// Function: sub_3960F30
// Address: 0x3960f30
//
__int64 __fastcall sub_3960F30(__int64 a1, int a2, void *a3, size_t a4, void *a5, size_t a6)
{
  __int64 v6; // rcx
  __int64 v7; // r12
  __int64 v8; // rbx
  void *v9; // rax
  __int64 v10; // r9
  __int64 result; // rax
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // [rsp+8h] [rbp-88h]
  void *s2; // [rsp+10h] [rbp-80h] BYREF
  size_t n; // [rsp+18h] [rbp-78h]
  _QWORD v17[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v18; // [rsp+30h] [rbp-60h]
  _QWORD v19[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v20; // [rsp+50h] [rbp-40h]

  if ( *(_QWORD *)(*(_QWORD *)(a1 + 200) + 32LL) )
  {
    s2 = a5;
    n = a6;
  }
  else
  {
    s2 = a3;
    n = a4;
  }
  v6 = *(unsigned int *)(a1 + 216);
  if ( *(_DWORD *)(a1 + 216) )
  {
    v7 = 0;
    v8 = *(_QWORD *)(a1 + 208);
    while ( 1 )
    {
      if ( *(_QWORD *)(v8 + 8) == n )
      {
        if ( !n )
          break;
        v14 = v6;
        v13 = memcmp(*(const void **)v8, s2, n);
        v6 = v14;
        if ( !v13 )
          break;
      }
      ++v7;
      v8 += 56;
      if ( v6 == v7 )
        goto LABEL_7;
    }
    v12 = *(_QWORD *)(v8 + 40);
  }
  else
  {
LABEL_7:
    v9 = sub_16E8CB0();
    v20 = 770;
    v18 = 1283;
    v17[0] = "Cannot find option named '";
    v17[1] = &s2;
    v19[0] = v17;
    v19[1] = "'!";
    result = sub_16B1F90(a1, (__int64)v19, 0, 0, (__int64)v9, v10);
    if ( (_BYTE)result )
      return result;
    v12 = 0;
  }
  *(_QWORD *)(a1 + 160) = v12;
  *(_DWORD *)(a1 + 16) = a2;
  return 0;
}
