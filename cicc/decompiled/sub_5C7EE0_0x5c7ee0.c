// Function: sub_5C7EE0
// Address: 0x5c7ee0
//
void __fastcall sub_5C7EE0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, _DWORD *a6)
{
  __int64 v9; // rcx
  _DWORD *v10; // [rsp+8h] [rbp-48h]
  __int64 v11; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v12[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( a4 && (*(_DWORD *)(a4 + 176) & 0x11000) == 0x1000 )
  {
    v10 = a6;
    sub_892370(a4, v12, &v11);
    v9 = 0;
    if ( (*(_BYTE *)(a4 + 89) & 4) != 0 )
      v9 = *(_QWORD *)(*(_QWORD *)(a4 + 40) + 32LL);
    sub_5C7EE0(a1, v12[0], v11, v9, a5);
    a6 = v10;
  }
  if ( !*a6 )
  {
    if ( a3 )
      *(_QWORD *)(a1 + 40) = sub_8A2270(*(_QWORD *)(a1 + 40), a3, a2, (int)a1 + 24, 0, (_DWORD)a6, a5);
  }
}
