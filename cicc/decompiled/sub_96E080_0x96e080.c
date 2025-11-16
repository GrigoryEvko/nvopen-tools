// Function: sub_96E080
// Address: 0x96e080
//
__int64 __fastcall sub_96E080(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  bool v8; // cc
  __int64 result; // rax
  __int64 v10; // rax
  __int16 v11; // dx
  unsigned int v12; // eax
  _QWORD *v13; // r8
  __int64 v14; // rcx
  unsigned __int8 v16; // [rsp+8h] [rbp-48h]
  unsigned __int8 v17; // [rsp+8h] [rbp-48h]
  __int64 v18; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-38h]

  while ( 1 )
  {
    if ( a5 )
      *a5 = 0;
    if ( *(_BYTE *)a1 <= 3u )
    {
      *a2 = a1;
      v19 = sub_AE43F0(a4, *(_QWORD *)(a1 + 8));
      if ( v19 <= 0x40 )
        goto LABEL_5;
      goto LABEL_12;
    }
    *a2 = 0;
    if ( *(_BYTE *)a1 == 6 )
    {
      if ( a5 )
        *a5 = a1;
      v10 = *(_QWORD *)(a1 - 32);
      *a2 = v10;
      v19 = sub_AE43F0(a4, *(_QWORD *)(v10 + 8));
      if ( v19 <= 0x40 )
      {
LABEL_5:
        v8 = *(_DWORD *)(a3 + 8) <= 0x40u;
        v18 = 0;
        if ( v8 )
        {
LABEL_6:
          *(_QWORD *)a3 = v18;
          *(_DWORD *)(a3 + 8) = v19;
          return 1;
        }
LABEL_17:
        if ( *(_QWORD *)a3 )
          j_j___libc_free_0_0(*(_QWORD *)a3);
        goto LABEL_6;
      }
LABEL_12:
      sub_C43690(&v18, 0, 0);
      if ( *(_DWORD *)(a3 + 8) <= 0x40u )
        goto LABEL_6;
      goto LABEL_17;
    }
    if ( *(_BYTE *)a1 != 5 )
      return 0;
    v11 = *(_WORD *)(a1 + 2);
    if ( ((v11 - 47) & 0xFFFD) != 0 )
      break;
    a1 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  }
  result = 0;
  if ( v11 == 34 )
  {
    v12 = sub_AE43F0(a4, *(_QWORD *)(a1 + 8));
    v13 = a5;
    v19 = v12;
    if ( v12 > 0x40 )
    {
      sub_C43690(&v18, 0, 0);
      v13 = a5;
    }
    else
    {
      v18 = 0;
    }
    if ( (unsigned __int8)sub_96E080(*(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)), a2, &v18, a4, v13)
      && (result = sub_BB6360(a1, a4, &v18, 0, 0), (_BYTE)result) )
    {
      if ( *(_DWORD *)(a3 + 8) > 0x40u || v19 > 0x40 )
      {
        v16 = result;
        sub_C43990(a3, &v18);
        result = v16;
        goto LABEL_28;
      }
      v14 = v18;
      *(_DWORD *)(a3 + 8) = v19;
      *(_QWORD *)a3 = v14;
    }
    else
    {
      result = 0;
LABEL_28:
      if ( v19 > 0x40 && v18 )
      {
        v17 = result;
        j_j___libc_free_0_0(v18);
        return v17;
      }
    }
  }
  return result;
}
