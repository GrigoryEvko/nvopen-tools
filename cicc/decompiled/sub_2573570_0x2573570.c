// Function: sub_2573570
// Address: 0x2573570
//
__int64 __fastcall sub_2573570(__int64 a1, __int64 *a2, _DWORD *a3)
{
  unsigned int v3; // r14d
  __int64 v4; // r15
  __int64 *v5; // rax
  __int64 *v6; // rdx
  __int64 **v7; // rax
  __int64 **v8; // rax
  __int64 *v10; // rax
  __int64 **v12; // [rsp+18h] [rbp-58h] BYREF
  __int64 *v13[10]; // [rsp+20h] [rbp-50h] BYREF

  v3 = *(unsigned __int8 *)(a1 + 97);
  if ( (_BYTE)v3 )
  {
    v4 = a1 + 168;
    if ( !a2[2] )
      goto LABEL_6;
    v5 = (__int64 *)a2[1];
    v6 = (__int64 *)*a2;
    v13[2] = 0;
    v13[0] = v6;
    v13[1] = v5;
    v13[3] = 0;
    v12 = v13;
    v7 = sub_25678E0(a1 + 168, (__int64 *)&v12);
    if ( !v7 )
      goto LABEL_6;
    if ( v7 != (__int64 **)(*(_QWORD *)(a1 + 176) + 8LL * *(unsigned int *)(a1 + 192)) && !*((_DWORD *)*v7 + 6) )
    {
      *a3 = 0;
    }
    else
    {
LABEL_6:
      v13[0] = a2;
      v8 = sub_25678E0(v4, (__int64 *)v13);
      if ( !v8 || v8 == (__int64 **)(*(_QWORD *)(a1 + 176) + 8LL * *(unsigned int *)(a1 + 192)) )
      {
        v12 = (__int64 **)a2;
        if ( !sub_2567AA0(v4, (__int64 *)&v12, v13) )
        {
          v10 = sub_2573440(v4, (__int64 *)&v12, v13[0]);
          *v10 = (__int64)v12;
        }
        return 0;
      }
      else
      {
        *a3 = *((_DWORD *)*v8 + 6);
      }
    }
  }
  else
  {
    *a3 = 1;
    return 1;
  }
  return v3;
}
