// Function: sub_1C1D500
// Address: 0x1c1d500
//
void __fastcall sub_1C1D500(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  char v8; // al
  _BOOL8 v9; // rcx
  size_t v10; // rdx
  char v11; // [rsp+7h] [rbp-39h] BYREF
  _QWORD v12[7]; // [rsp+8h] [rbp-38h] BYREF

  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v9 = 0;
  if ( v8 )
  {
    v10 = *(_QWORD *)(a3 + 8) - *(_QWORD *)a3;
    if ( v10 == *(_QWORD *)(a4 + 8) - *(_QWORD *)a4 )
    {
      v9 = 1;
      if ( v10 )
        v9 = memcmp(*(const void **)a3, *(const void **)a4, v10) == 0;
    }
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         a2,
         a5,
         v9,
         &v11,
         v12) )
  {
    sub_1C1D3E0(a1, (__int64 *)a3);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v12[0]);
  }
  else if ( v11 )
  {
    sub_1C13A10(a3, (char **)a4);
  }
}
