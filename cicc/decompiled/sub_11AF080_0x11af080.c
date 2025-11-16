// Function: sub_11AF080
// Address: 0x11af080
//
__int64 __fastcall sub_11AF080(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  int v4; // edx
  __int64 v5; // rbx
  _BYTE *v6; // r13
  __int64 v7; // r15
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned int v11; // r14d
  bool v12; // al
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rdx
  _BYTE *v16; // rax
  unsigned int v17; // ebx
  unsigned int v18; // r14d
  __int64 v19; // rax
  char v20; // [rsp+8h] [rbp-38h]
  int v21; // [rsp+8h] [rbp-38h]
  int v22; // [rsp+Ch] [rbp-34h]

  v4 = *a2;
  v5 = *((_QWORD *)a2 - 8);
  v6 = (_BYTE *)*((_QWORD *)a2 - 4);
  v7 = *((_QWORD *)a2 + 1);
  switch ( v4 )
  {
    case '6':
      if ( *v6 <= 0x15u && *v6 != 5 && !(unsigned __int8)sub_AD6CA0(*((_QWORD *)a2 - 4)) )
      {
        v9 = sub_AD64C0(v7, 1, 0);
        v10 = sub_96E6C0(0x19u, v9, v6, a3);
        *(_DWORD *)a1 = 17;
        *(_QWORD *)(a1 + 8) = v5;
        *(_QWORD *)(a1 + 16) = v10;
        return a1;
      }
      break;
    case ':':
      if ( (a2[1] & 2) != 0 )
      {
        *(_DWORD *)a1 = 13;
        *(_QWORD *)(a1 + 8) = v5;
        *(_QWORD *)(a1 + 16) = v6;
        return a1;
      }
      break;
    case ',':
      if ( *(_BYTE *)v5 == 17 )
      {
        v11 = *(_DWORD *)(v5 + 32);
        if ( v11 <= 0x40 )
          v12 = *(_QWORD *)(v5 + 24) == 0;
        else
          v12 = v11 == (unsigned int)sub_C444A0(v5 + 24);
      }
      else
      {
        v14 = *(_QWORD *)(v5 + 8);
        v15 = (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17;
        if ( (unsigned int)v15 > 1 || *(_BYTE *)v5 > 0x15u )
          break;
        v16 = sub_AD7630(v5, 0, v15);
        if ( !v16 || *v16 != 17 )
        {
          if ( *(_BYTE *)(v14 + 8) == 17 )
          {
            v22 = *(_DWORD *)(v14 + 32);
            if ( v22 )
            {
              v20 = 0;
              v18 = 0;
              while ( 1 )
              {
                v19 = sub_AD69F0((unsigned __int8 *)v5, v18);
                if ( !v19 )
                  break;
                if ( *(_BYTE *)v19 != 13 )
                {
                  if ( *(_BYTE *)v19 != 17 )
                    goto LABEL_4;
                  if ( *(_DWORD *)(v19 + 32) <= 0x40u )
                  {
                    if ( *(_QWORD *)(v19 + 24) )
                      goto LABEL_4;
                  }
                  else
                  {
                    v21 = *(_DWORD *)(v19 + 32);
                    if ( v21 != (unsigned int)sub_C444A0(v19 + 24) )
                      goto LABEL_4;
                  }
                  v20 = 1;
                }
                if ( v22 == ++v18 )
                {
                  if ( v20 )
                    goto LABEL_14;
                  goto LABEL_4;
                }
              }
            }
          }
          break;
        }
        v17 = *((_DWORD *)v16 + 8);
        if ( v17 <= 0x40 )
        {
          if ( *((_QWORD *)v16 + 3) )
            break;
          goto LABEL_14;
        }
        v12 = v17 == (unsigned int)sub_C444A0((__int64)(v16 + 24));
      }
      if ( !v12 )
        break;
LABEL_14:
      v13 = sub_AD62B0(v7);
      *(_DWORD *)a1 = 17;
      *(_QWORD *)(a1 + 8) = v6;
      *(_QWORD *)(a1 + 16) = v13;
      return a1;
  }
LABEL_4:
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  return a1;
}
