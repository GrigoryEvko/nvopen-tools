// Function: sub_14A4410
// Address: 0x14a4410
//
__int64 __fastcall sub_14A4410(__int64 a1, __int64 a2)
{
  int v3; // eax
  char v4; // cl
  __int64 v5; // rdx
  __int64 v6; // rdi
  __int64 v8; // rdx
  int v9; // eax
  int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // r13
  __int64 v13; // rdi
  __int64 v14; // r14
  int v15; // eax
  int v16; // eax
  __int64 v17; // r15
  __int64 v18; // rbx
  __int64 v19; // rcx
  __int64 v20; // rdi
  int v21; // eax
  int v22; // eax
  int v23; // eax
  int v24; // eax
  int v25; // eax
  int v26; // eax
  int v27; // eax
  __int64 v28; // [rsp+10h] [rbp-60h] BYREF
  __int64 v29; // [rsp+18h] [rbp-58h] BYREF
  _QWORD *v30[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v31[8]; // [rsp+30h] [rbp-40h] BYREF

  v3 = *(unsigned __int8 *)(a2 + 16);
  v4 = *(_BYTE *)(a2 + 16);
  if ( (unsigned int)(v3 - 35) > 0x11
    || (v5 = *(_QWORD *)(a2 - 48)) == 0
    || (v6 = *(_QWORD *)(a2 - 24), v28 = *(_QWORD *)(a2 - 48), !v6) )
  {
    if ( v4 != 79 )
      goto LABEL_12;
    v8 = *(_QWORD *)(a2 - 72);
    v9 = *(unsigned __int8 *)(v8 + 16);
    if ( (_BYTE)v9 != 75 )
      goto LABEL_8;
    v11 = *(_QWORD *)(a2 - 48);
    v12 = *(_QWORD *)(v8 - 48);
    v13 = *(_QWORD *)(a2 - 24);
    v14 = *(_QWORD *)(v8 - 24);
    if ( v11 == v12 && v13 == v14 )
    {
      if ( (*(_WORD *)(v8 + 18) & 0x7FFFu) - 40 > 1 || !v12 )
        goto LABEL_29;
    }
    else
    {
      if ( v11 != v14 )
        goto LABEL_9;
      if ( v13 != v12 )
        goto LABEL_43;
      if ( v11 != v12 )
      {
        v15 = sub_15FF0F0(*(_WORD *)(v8 + 18) & 0x7FFF);
        v8 = *(_QWORD *)(a2 - 72);
        if ( (unsigned int)(v15 - 40) <= 1 && v12 )
        {
          v28 = v12;
          v9 = *(unsigned __int8 *)(v8 + 16);
          if ( v14 )
            goto LABEL_18;
        }
        else
        {
          LOBYTE(v9) = *(_BYTE *)(v8 + 16);
        }
        if ( (_BYTE)v9 != 75 )
          goto LABEL_8;
        v11 = *(_QWORD *)(a2 - 48);
        v13 = *(_QWORD *)(a2 - 24);
LABEL_27:
        v14 = *(_QWORD *)(v8 - 24);
        v12 = *(_QWORD *)(v8 - 48);
        if ( v13 == v14 && v12 == v11 )
          goto LABEL_29;
        goto LABEL_43;
      }
      if ( (*(_WORD *)(v8 + 18) & 0x7FFFu) - 40 > 1 || !v11 )
      {
LABEL_43:
        if ( v11 != v14 || v13 != v12 )
          goto LABEL_9;
        if ( v11 != v12 )
        {
          v16 = sub_15FF0F0(*(_WORD *)(v8 + 18) & 0x7FFF);
          v8 = *(_QWORD *)(a2 - 72);
          goto LABEL_30;
        }
LABEL_29:
        v16 = *(unsigned __int16 *)(v8 + 18);
        BYTE1(v16) &= ~0x80u;
LABEL_30:
        if ( (unsigned int)(v16 - 38) > 1 || !v12 )
        {
          LOBYTE(v9) = *(_BYTE *)(v8 + 16);
          goto LABEL_8;
        }
        v28 = v12;
        v9 = *(unsigned __int8 *)(v8 + 16);
        if ( v14 )
        {
LABEL_18:
          *(_BYTE *)(a1 + 32) = 1;
          *(_DWORD *)a1 = v9 - 24;
          *(_QWORD *)(a1 + 8) = v12;
          *(_QWORD *)(a1 + 16) = v14;
          *(_DWORD *)(a1 + 24) = 2;
          return a1;
        }
LABEL_8:
        if ( (_BYTE)v9 != 76 )
          goto LABEL_9;
        v17 = *(_QWORD *)(v8 - 48);
        v18 = *(_QWORD *)(v8 - 24);
        v19 = *(_QWORD *)(a2 - 48);
        v20 = *(_QWORD *)(a2 - 24);
        v12 = v17;
        v14 = v18;
        if ( v19 == v17 && v20 == v18 )
        {
          v21 = *(unsigned __int16 *)(v8 + 18);
          BYTE1(v21) &= ~0x80u;
          if ( (unsigned int)(v21 - 4) > 1 || !v17 )
          {
            v12 = *(_QWORD *)(v8 - 48);
            v14 = *(_QWORD *)(v8 - 24);
            goto LABEL_55;
          }
        }
        else
        {
          if ( v19 != v18 || v20 != v17 )
            goto LABEL_81;
          if ( v19 != v17 )
          {
            v22 = sub_15FF0F0(*(_WORD *)(v8 + 18) & 0x7FFF);
            v8 = *(_QWORD *)(a2 - 72);
            if ( (unsigned int)(v22 - 4) > 1 || !v17 )
              goto LABEL_52;
            goto LABEL_39;
          }
          if ( (*(_WORD *)(v8 + 18) & 0x7FFFu) - 4 > 1 || !v19 )
          {
LABEL_81:
            v12 = *(_QWORD *)(v8 - 48);
            v14 = *(_QWORD *)(v8 - 24);
LABEL_82:
            if ( v14 != v19 || v12 != v20 )
              goto LABEL_60;
            if ( v12 != v19 )
            {
              v26 = sub_15FF0F0(*(_WORD *)(v8 + 18) & 0x7FFF);
              v8 = *(_QWORD *)(a2 - 72);
              if ( (unsigned int)(v26 - 2) > 1 || !v12 )
              {
LABEL_58:
                if ( *(_BYTE *)(v8 + 16) != 76 )
                  goto LABEL_9;
                v19 = *(_QWORD *)(a2 - 48);
                v20 = *(_QWORD *)(a2 - 24);
LABEL_60:
                v12 = *(_QWORD *)(v8 - 48);
                v14 = *(_QWORD *)(v8 - 24);
                if ( v19 == v12 && v20 == v14 )
                {
                  v24 = *(unsigned __int16 *)(v8 + 18);
                }
                else
                {
                  if ( v19 != v14 || v20 != v12 )
                    goto LABEL_68;
                  v24 = *(unsigned __int16 *)(v8 + 18);
                  if ( v19 != v12 )
                  {
                    v27 = sub_15FF0F0(*(_WORD *)(v8 + 18) & 0x7FFF);
                    v8 = *(_QWORD *)(a2 - 72);
                    if ( (unsigned int)(v27 - 12) > 1 || !v12 )
                    {
LABEL_66:
                      if ( *(_BYTE *)(v8 + 16) != 76 )
                        goto LABEL_9;
                      v19 = *(_QWORD *)(a2 - 48);
                      v20 = *(_QWORD *)(a2 - 24);
LABEL_68:
                      v12 = *(_QWORD *)(v8 - 48);
                      v14 = *(_QWORD *)(v8 - 24);
                      if ( v19 == v12 && v20 == v14 )
                      {
                        v25 = *(unsigned __int16 *)(v8 + 18);
                      }
                      else
                      {
                        if ( v19 != v14 || v20 != v12 )
                          goto LABEL_9;
                        v25 = *(unsigned __int16 *)(v8 + 18);
                        if ( v19 != v12 )
                        {
                          v25 = sub_15FF0F0(*(_WORD *)(v8 + 18) & 0x7FFF);
                          goto LABEL_72;
                        }
                      }
                      BYTE1(v25) &= ~0x80u;
LABEL_72:
                      if ( (unsigned int)(v25 - 10) <= 1 )
                      {
                        if ( v12 )
                        {
                          v28 = v12;
                          if ( v14 )
                          {
                            v9 = *(unsigned __int8 *)(*(_QWORD *)(a2 - 72) + 16LL);
                            goto LABEL_18;
                          }
                        }
                      }
LABEL_9:
                      v30[0] = &v28;
                      v30[1] = &v29;
                      if ( sub_14A42B0(v30, a2) || (v31[0] = &v28, v31[1] = &v29, sub_14A4360(v31, a2)) )
                      {
                        v10 = *(unsigned __int8 *)(*(_QWORD *)(a2 - 72) + 16LL);
                        *(_BYTE *)(a1 + 32) = 1;
                        *(_DWORD *)(a1 + 24) = 3;
                        *(_DWORD *)a1 = v10 - 24;
                        *(_QWORD *)(a1 + 8) = v28;
                        *(_QWORD *)(a1 + 16) = v29;
                        return a1;
                      }
LABEL_12:
                      *(_BYTE *)(a1 + 32) = 0;
                      return a1;
                    }
LABEL_65:
                    v28 = v12;
                    if ( v14 )
                      goto LABEL_40;
                    goto LABEL_66;
                  }
                }
                BYTE1(v24) &= ~0x80u;
                if ( (unsigned int)(v24 - 12) > 1 || !v12 )
                  goto LABEL_68;
                goto LABEL_65;
              }
LABEL_57:
              v28 = v12;
              if ( v14 )
                goto LABEL_40;
              goto LABEL_58;
            }
LABEL_55:
            v23 = *(unsigned __int16 *)(v8 + 18);
            BYTE1(v23) &= ~0x80u;
            if ( (unsigned int)(v23 - 2) > 1 || !v12 )
              goto LABEL_60;
            goto LABEL_57;
          }
        }
LABEL_39:
        v28 = v17;
        if ( v18 )
        {
LABEL_40:
          v9 = *(unsigned __int8 *)(v8 + 16);
          goto LABEL_18;
        }
LABEL_52:
        if ( *(_BYTE *)(v8 + 16) != 76 )
          goto LABEL_9;
        v19 = *(_QWORD *)(a2 - 48);
        v12 = *(_QWORD *)(v8 - 48);
        v20 = *(_QWORD *)(a2 - 24);
        v14 = *(_QWORD *)(v8 - 24);
        if ( v19 == v12 && v20 == v14 )
          goto LABEL_55;
        goto LABEL_82;
      }
    }
    v28 = *(_QWORD *)(v8 - 48);
    if ( v14 )
      goto LABEL_18;
    goto LABEL_27;
  }
  *(_BYTE *)(a1 + 32) = 1;
  *(_DWORD *)a1 = v3 - 24;
  *(_QWORD *)(a1 + 8) = v5;
  *(_QWORD *)(a1 + 16) = v6;
  *(_DWORD *)(a1 + 24) = 1;
  return a1;
}
