// Function: sub_109D670
// Address: 0x109d670
//
__int64 __fastcall sub_109D670(_BYTE *a1, _QWORD *a2, __int64 a3)
{
  char v6; // al
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rdx
  _BYTE *v15; // rax
  __int64 v16; // rdx
  _BYTE *v17; // rax
  __int64 v18; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v19; // [rsp+8h] [rbp-28h]

  v6 = *a1;
  if ( *a1 != 46 )
  {
LABEL_2:
    if ( v6 == 54 )
    {
      v8 = *((_QWORD *)a1 - 8);
      if ( v8 )
      {
        *a2 = v8;
        v9 = *((_QWORD *)a1 - 4);
        v10 = v9 + 24;
        if ( *(_BYTE *)v9 == 17 )
        {
LABEL_6:
          v19 = *(_DWORD *)(v10 + 8);
          if ( v19 > 0x40 )
            sub_C43690((__int64)&v18, 1, 0);
          else
            v18 = 1;
          if ( *(_DWORD *)(a3 + 8) > 0x40u )
          {
            if ( *(_QWORD *)a3 )
              j_j___libc_free_0_0(*(_QWORD *)a3);
          }
          *(_QWORD *)a3 = v18;
          *(_DWORD *)(a3 + 8) = v19;
          sub_C47AC0(a3, v10);
          return 1;
        }
        v16 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v9 + 8) + 8LL) - 17;
        if ( (unsigned int)v16 <= 1 && *(_BYTE *)v9 <= 0x15u )
        {
          v17 = sub_AD7630(v9, 0, v16);
          if ( v17 )
          {
            if ( *v17 == 17 )
            {
              v10 = (__int64)(v17 + 24);
              goto LABEL_6;
            }
          }
        }
      }
    }
    return 0;
  }
  v11 = *((_QWORD *)a1 - 8);
  if ( !v11 )
    return 0;
  *a2 = v11;
  v12 = *((_QWORD *)a1 - 4);
  v13 = v12 + 24;
  if ( *(_BYTE *)v12 != 17 )
  {
    v14 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v12 + 8) + 8LL) - 17;
    if ( (unsigned int)v14 > 1
      || *(_BYTE *)v12 > 0x15u
      || (v15 = sub_AD7630(v12, 0, v14)) == 0
      || (v13 = (__int64)(v15 + 24), *v15 != 17) )
    {
      v6 = *a1;
      goto LABEL_2;
    }
  }
  if ( *(_DWORD *)(a3 + 8) > 0x40u || *(_DWORD *)(v13 + 8) > 0x40u )
  {
    sub_C43990(a3, v13);
    return 1;
  }
  else
  {
    *(_QWORD *)a3 = *(_QWORD *)v13;
    *(_DWORD *)(a3 + 8) = *(_DWORD *)(v13 + 8);
    return 1;
  }
}
