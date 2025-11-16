// Function: sub_1004E80
// Address: 0x1004e80
//
__int64 __fastcall sub_1004E80(__int64 **a1, __int64 a2)
{
  unsigned int v2; // r14d
  __int64 v3; // rax
  int v5; // r13d
  __int64 v6; // r13
  __int64 v7; // rdx
  _BYTE *v8; // rax
  unsigned int v9; // r15d
  __int64 v10; // r13
  int v11; // r14d
  int v12; // r13d
  char v13; // r14
  unsigned int v14; // r15d
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // rax
  int v18; // [rsp+8h] [rbp-38h]
  int v19; // [rsp+Ch] [rbp-34h]

  if ( *(_BYTE *)a2 == 17 )
  {
    v2 = *(_DWORD *)(a2 + 32);
    if ( v2 <= 0x40 )
    {
      v3 = *(_QWORD *)(a2 + 24);
      if ( !v3 )
        return 0;
      goto LABEL_4;
    }
    v5 = sub_C445E0(a2 + 24);
    if ( !v5 || v2 != (unsigned int)sub_C444A0(a2 + 24) + v5 )
      return 0;
  }
  else
  {
    v6 = *(_QWORD *)(a2 + 8);
    v7 = (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17;
    if ( (unsigned int)v7 > 1 || *(_BYTE *)a2 > 0x15u )
      return 0;
    v8 = sub_AD7630(a2, 0, v7);
    if ( !v8 || *v8 != 17 )
    {
      if ( *(_BYTE *)(v6 + 8) == 17 )
      {
        v12 = *(_DWORD *)(v6 + 32);
        if ( v12 )
        {
          v13 = 0;
          v14 = 0;
          while ( 1 )
          {
            v15 = sub_AD69F0((unsigned __int8 *)a2, v14);
            if ( !v15 )
              break;
            if ( *(_BYTE *)v15 != 13 )
            {
              if ( *(_BYTE *)v15 != 17 )
                return 0;
              if ( *(_DWORD *)(v15 + 32) <= 0x40u )
              {
                v17 = *(_QWORD *)(v15 + 24);
                if ( !v17 || (v17 & (v17 + 1)) != 0 )
                  return 0;
              }
              else
              {
                v16 = v15 + 24;
                v18 = *(_DWORD *)(v15 + 32);
                v19 = sub_C445E0(v15 + 24);
                if ( !v19 || v18 != (unsigned int)sub_C444A0(v16) + v19 )
                  return 0;
              }
              v13 = 1;
            }
            if ( v12 == ++v14 )
            {
              if ( v13 )
                goto LABEL_5;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v9 = *((_DWORD *)v8 + 8);
    if ( v9 <= 0x40 )
    {
      v3 = *((_QWORD *)v8 + 3);
      if ( !v3 )
        return 0;
LABEL_4:
      if ( (v3 & (v3 + 1)) == 0 )
        goto LABEL_5;
      return 0;
    }
    v10 = (__int64)(v8 + 24);
    v11 = sub_C445E0((__int64)(v8 + 24));
    if ( !v11 || v9 != (unsigned int)sub_C444A0(v10) + v11 )
      return 0;
  }
LABEL_5:
  if ( *a1 )
    **a1 = a2;
  return 1;
}
