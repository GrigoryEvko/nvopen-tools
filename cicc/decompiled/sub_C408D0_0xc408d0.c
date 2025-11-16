// Function: sub_C408D0
// Address: 0xc408d0
//
__int64 __fastcall sub_C408D0(__int64 a1, void **a2)
{
  bool v2; // al
  unsigned int v3; // r12d
  _DWORD *v5; // [rsp+0h] [rbp-C0h]
  __int64 v6[4]; // [rsp+10h] [rbp-B0h] BYREF
  void *v7[4]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v8[4]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v9[10]; // [rsp+70h] [rbp-50h] BYREF

  v2 = (*(_BYTE *)(a1 + 20) & 6) != 0;
  v3 = v2 && (*(_BYTE *)(a1 + 20) & 7) != 3;
  if ( v2 && (*(_BYTE *)(a1 + 20) & 7) != 3 )
  {
    if ( (unsigned int)sub_C34270(a1) == *(_DWORD *)(*(_QWORD *)a1 + 8LL) - 1 )
    {
      sub_C36740((__int64)v6, *(_QWORD *)a1, 1);
      if ( (unsigned int)sub_C3B6C0((__int64)v6, a1, 1) || sub_C33940((__int64)v6) )
      {
        v3 = 0;
      }
      else if ( a2 )
      {
        v5 = *(_DWORD **)a1;
        sub_C33EB0(v8, v6);
        sub_C338E0((__int64)v9, (__int64)v8);
        sub_C407B0(v7, v9, v5);
        sub_C338F0((__int64)v9);
        sub_C3C870(a2, v7);
        sub_91D830(v7);
        sub_C338F0((__int64)v8);
      }
      sub_C338F0((__int64)v6);
    }
    else
    {
      return 0;
    }
  }
  return v3;
}
