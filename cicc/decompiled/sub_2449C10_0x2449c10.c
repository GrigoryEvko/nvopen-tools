// Function: sub_2449C10
// Address: 0x2449c10
//
void __fastcall sub_2449C10(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned int v4; // eax
  _BYTE *v5; // rsi
  unsigned __int8 **v6; // rax
  unsigned __int8 *v7; // rax
  _BYTE *v8; // rsi
  _BYTE *v9; // rsi
  unsigned __int8 **v10; // rax
  unsigned __int8 *v11; // rax
  _BYTE *v12; // rsi
  unsigned __int8 **v13; // rax
  unsigned __int8 *v14; // rax
  _BYTE *v15; // rsi
  unsigned __int8 **v16; // rax
  unsigned __int8 *v17; // rax
  _BYTE *v18; // rsi
  unsigned __int8 **v19; // rax
  unsigned __int8 *v20; // rax
  _BYTE *v21; // rsi
  unsigned __int8 **v22; // rax
  unsigned __int8 *v23; // rax
  _BYTE *v24; // rsi
  unsigned __int8 **v25; // rax
  unsigned __int8 *v26; // rax
  _BYTE *v27; // rsi
  unsigned __int8 **v28; // rax
  unsigned __int8 *v29; // rax
  _BYTE *v30; // rsi
  unsigned __int8 **v31; // rax
  unsigned __int8 *v32; // rax
  _BYTE *v33; // rsi
  unsigned __int8 **v34; // rax
  unsigned __int8 *v35; // rax
  _BYTE *v36; // rsi
  unsigned __int8 **v37; // rax
  unsigned __int8 *v38; // rax
  _BYTE *v39; // rsi
  unsigned __int8 **v40; // rax
  unsigned __int8 *v41; // rax
  _BYTE *v42; // rsi
  unsigned __int8 **v43; // rax
  unsigned __int8 *v44; // rax
  _BYTE *v45; // rsi
  unsigned __int8 **v46; // rax
  unsigned __int8 *v47; // rax
  _QWORD v48[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = *(_QWORD *)(a2 - 32);
  if ( !v3 || *(_BYTE *)v3 || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a2 + 80) )
    goto LABEL_4;
  v4 = *(_DWORD *)(v3 + 36);
  if ( v4 > 0xF5 )
  {
    switch ( v4 )
    {
      case 0x176u:
        if ( sub_B491E0(a2) )
        {
          v48[0] = a2;
          v24 = *(_BYTE **)(a1 + 8);
          if ( v24 == *(_BYTE **)(a1 + 16) )
          {
            sub_2445670(a1, v24, v48);
          }
          else
          {
            if ( v24 )
            {
              *(_QWORD *)v24 = a2;
              v24 = *(_BYTE **)(a1 + 8);
            }
            v24 += 8;
            *(_QWORD *)(a1 + 8) = v24;
          }
          if ( *(_DWORD *)(a1 + 48) == 1 && sub_B491E0(a2) )
          {
            v25 = *(unsigned __int8 ***)(a2 - 32);
            if ( *(_BYTE *)v25 == 61 )
            {
              v26 = sub_BD4070(*(v25 - 4), (__int64)v24);
              if ( v26 )
              {
                if ( *v26 > 0x1Cu )
                {
                  v48[0] = v26;
                  v8 = *(_BYTE **)(a1 + 32);
                  if ( v8 == *(_BYTE **)(a1 + 40) )
                    goto LABEL_228;
                  if ( v8 )
                    *(_QWORD *)v8 = v26;
                  *(_QWORD *)(a1 + 32) += 8LL;
                }
              }
            }
          }
        }
        return;
      case 0x177u:
        if ( sub_B491E0(a2) )
        {
          v48[0] = a2;
          v9 = *(_BYTE **)(a1 + 8);
          if ( v9 == *(_BYTE **)(a1 + 16) )
          {
            sub_2445670(a1, v9, v48);
          }
          else
          {
            if ( v9 )
            {
              *(_QWORD *)v9 = a2;
              v9 = *(_BYTE **)(a1 + 8);
            }
            v9 += 8;
            *(_QWORD *)(a1 + 8) = v9;
          }
          if ( *(_DWORD *)(a1 + 48) == 1 && sub_B491E0(a2) )
          {
            v10 = *(unsigned __int8 ***)(a2 - 32);
            if ( *(_BYTE *)v10 == 61 )
            {
              v11 = sub_BD4070(*(v10 - 4), (__int64)v9);
              if ( v11 )
              {
                if ( *v11 > 0x1Cu )
                {
                  v48[0] = v11;
                  v8 = *(_BYTE **)(a1 + 32);
                  if ( v8 == *(_BYTE **)(a1 + 40) )
                    goto LABEL_228;
                  if ( v8 )
                    *(_QWORD *)v8 = v11;
                  *(_QWORD *)(a1 + 32) += 8LL;
                }
              }
            }
          }
        }
        return;
      case 0x175u:
        if ( sub_B491E0(a2) )
        {
          v48[0] = a2;
          v18 = *(_BYTE **)(a1 + 8);
          if ( v18 == *(_BYTE **)(a1 + 16) )
          {
            sub_2445670(a1, v18, v48);
          }
          else
          {
            if ( v18 )
            {
              *(_QWORD *)v18 = a2;
              v18 = *(_BYTE **)(a1 + 8);
            }
            v18 += 8;
            *(_QWORD *)(a1 + 8) = v18;
          }
          if ( *(_DWORD *)(a1 + 48) == 1 && sub_B491E0(a2) )
          {
            v19 = *(unsigned __int8 ***)(a2 - 32);
            if ( *(_BYTE *)v19 == 61 )
            {
              v20 = sub_BD4070(*(v19 - 4), (__int64)v18);
              if ( v20 )
              {
                if ( *v20 > 0x1Cu )
                {
                  v48[0] = v20;
                  v8 = *(_BYTE **)(a1 + 32);
                  if ( v8 == *(_BYTE **)(a1 + 40) )
                    goto LABEL_228;
                  if ( v8 )
                    *(_QWORD *)v8 = v20;
                  *(_QWORD *)(a1 + 32) += 8LL;
                }
              }
            }
          }
        }
        return;
    }
    goto LABEL_115;
  }
  if ( v4 > 0xED )
  {
    switch ( v4 )
    {
      case 0xEEu:
        if ( !sub_B491E0(a2) )
          return;
        v48[0] = a2;
        v45 = *(_BYTE **)(a1 + 8);
        if ( v45 == *(_BYTE **)(a1 + 16) )
        {
          sub_2445670(a1, v45, v48);
        }
        else
        {
          if ( v45 )
          {
            *(_QWORD *)v45 = a2;
            v45 = *(_BYTE **)(a1 + 8);
          }
          v45 += 8;
          *(_QWORD *)(a1 + 8) = v45;
        }
        if ( *(_DWORD *)(a1 + 48) != 1 )
          return;
        if ( !sub_B491E0(a2) )
          return;
        v46 = *(unsigned __int8 ***)(a2 - 32);
        if ( *(_BYTE *)v46 != 61 )
          return;
        v47 = sub_BD4070(*(v46 - 4), (__int64)v45);
        if ( !v47 || *v47 <= 0x1Cu )
          return;
        v48[0] = v47;
        v8 = *(_BYTE **)(a1 + 32);
        if ( v8 == *(_BYTE **)(a1 + 40) )
          goto LABEL_228;
        if ( v8 )
          *(_QWORD *)v8 = v47;
        *(_QWORD *)(a1 + 32) += 8LL;
        break;
      case 0xF0u:
        if ( !sub_B491E0(a2) )
          return;
        v48[0] = a2;
        v42 = *(_BYTE **)(a1 + 8);
        if ( v42 == *(_BYTE **)(a1 + 16) )
        {
          sub_2445670(a1, v42, v48);
        }
        else
        {
          if ( v42 )
          {
            *(_QWORD *)v42 = a2;
            v42 = *(_BYTE **)(a1 + 8);
          }
          v42 += 8;
          *(_QWORD *)(a1 + 8) = v42;
        }
        if ( *(_DWORD *)(a1 + 48) != 1 )
          return;
        if ( !sub_B491E0(a2) )
          return;
        v43 = *(unsigned __int8 ***)(a2 - 32);
        if ( *(_BYTE *)v43 != 61 )
          return;
        v44 = sub_BD4070(*(v43 - 4), (__int64)v42);
        if ( !v44 || *v44 <= 0x1Cu )
          return;
        v48[0] = v44;
        v8 = *(_BYTE **)(a1 + 32);
        if ( v8 == *(_BYTE **)(a1 + 40) )
          goto LABEL_228;
        if ( v8 )
          *(_QWORD *)v8 = v44;
        *(_QWORD *)(a1 + 32) += 8LL;
        break;
      case 0xF1u:
        if ( !sub_B491E0(a2) )
          return;
        v48[0] = a2;
        v39 = *(_BYTE **)(a1 + 8);
        if ( v39 == *(_BYTE **)(a1 + 16) )
        {
          sub_2445670(a1, v39, v48);
        }
        else
        {
          if ( v39 )
          {
            *(_QWORD *)v39 = a2;
            v39 = *(_BYTE **)(a1 + 8);
          }
          v39 += 8;
          *(_QWORD *)(a1 + 8) = v39;
        }
        if ( *(_DWORD *)(a1 + 48) != 1 )
          return;
        if ( !sub_B491E0(a2) )
          return;
        v40 = *(unsigned __int8 ***)(a2 - 32);
        if ( *(_BYTE *)v40 != 61 )
          return;
        v41 = sub_BD4070(*(v40 - 4), (__int64)v39);
        if ( !v41 || *v41 <= 0x1Cu )
          return;
        v48[0] = v41;
        v8 = *(_BYTE **)(a1 + 32);
        if ( v8 == *(_BYTE **)(a1 + 40) )
          goto LABEL_228;
        if ( v8 )
          *(_QWORD *)v8 = v41;
        *(_QWORD *)(a1 + 32) += 8LL;
        break;
      case 0xF3u:
        if ( !sub_B491E0(a2) )
          return;
        v48[0] = a2;
        v36 = *(_BYTE **)(a1 + 8);
        if ( v36 == *(_BYTE **)(a1 + 16) )
        {
          sub_2445670(a1, v36, v48);
        }
        else
        {
          if ( v36 )
          {
            *(_QWORD *)v36 = a2;
            v36 = *(_BYTE **)(a1 + 8);
          }
          v36 += 8;
          *(_QWORD *)(a1 + 8) = v36;
        }
        if ( *(_DWORD *)(a1 + 48) != 1 )
          return;
        if ( !sub_B491E0(a2) )
          return;
        v37 = *(unsigned __int8 ***)(a2 - 32);
        if ( *(_BYTE *)v37 != 61 )
          return;
        v38 = sub_BD4070(*(v37 - 4), (__int64)v36);
        if ( !v38 || *v38 <= 0x1Cu )
          return;
        v48[0] = v38;
        v8 = *(_BYTE **)(a1 + 32);
        if ( v8 == *(_BYTE **)(a1 + 40) )
          goto LABEL_228;
        if ( v8 )
          *(_QWORD *)v8 = v38;
        *(_QWORD *)(a1 + 32) += 8LL;
        break;
      case 0xF5u:
        if ( !sub_B491E0(a2) )
          return;
        v48[0] = a2;
        v33 = *(_BYTE **)(a1 + 8);
        if ( v33 == *(_BYTE **)(a1 + 16) )
        {
          sub_2445670(a1, v33, v48);
        }
        else
        {
          if ( v33 )
          {
            *(_QWORD *)v33 = a2;
            v33 = *(_BYTE **)(a1 + 8);
          }
          v33 += 8;
          *(_QWORD *)(a1 + 8) = v33;
        }
        if ( *(_DWORD *)(a1 + 48) != 1 )
          return;
        if ( !sub_B491E0(a2) )
          return;
        v34 = *(unsigned __int8 ***)(a2 - 32);
        if ( *(_BYTE *)v34 != 61 )
          return;
        v35 = sub_BD4070(*(v34 - 4), (__int64)v33);
        if ( !v35 || *v35 <= 0x1Cu )
          return;
        v48[0] = v35;
        v8 = *(_BYTE **)(a1 + 32);
        if ( v8 == *(_BYTE **)(a1 + 40) )
          goto LABEL_228;
        if ( v8 )
          *(_QWORD *)v8 = v35;
        *(_QWORD *)(a1 + 32) += 8LL;
        break;
      default:
        goto LABEL_115;
    }
    return;
  }
  if ( v4 == 70 )
  {
    if ( sub_B491E0(a2) )
    {
      v48[0] = a2;
      v30 = *(_BYTE **)(a1 + 8);
      if ( v30 == *(_BYTE **)(a1 + 16) )
      {
        sub_2445670(a1, v30, v48);
      }
      else
      {
        if ( v30 )
        {
          *(_QWORD *)v30 = a2;
          v30 = *(_BYTE **)(a1 + 8);
        }
        v30 += 8;
        *(_QWORD *)(a1 + 8) = v30;
      }
      if ( *(_DWORD *)(a1 + 48) == 1 && sub_B491E0(a2) )
      {
        v31 = *(unsigned __int8 ***)(a2 - 32);
        if ( *(_BYTE *)v31 == 61 )
        {
          v32 = sub_BD4070(*(v31 - 4), (__int64)v30);
          if ( v32 )
          {
            if ( *v32 > 0x1Cu )
            {
              v48[0] = v32;
              v8 = *(_BYTE **)(a1 + 32);
              if ( v8 == *(_BYTE **)(a1 + 40) )
                goto LABEL_228;
              if ( v8 )
                *(_QWORD *)v8 = v32;
              *(_QWORD *)(a1 + 32) += 8LL;
            }
          }
        }
      }
    }
    return;
  }
  if ( v4 <= 0x46 )
  {
    if ( !v4 )
    {
LABEL_4:
      if ( sub_B491E0(a2) )
      {
        v48[0] = a2;
        v15 = *(_BYTE **)(a1 + 8);
        if ( v15 == *(_BYTE **)(a1 + 16) )
        {
          sub_2445670(a1, v15, v48);
        }
        else
        {
          if ( v15 )
          {
            *(_QWORD *)v15 = a2;
            v15 = *(_BYTE **)(a1 + 8);
          }
          v15 += 8;
          *(_QWORD *)(a1 + 8) = v15;
        }
        if ( *(_DWORD *)(a1 + 48) == 1 && sub_B491E0(a2) )
        {
          v16 = *(unsigned __int8 ***)(a2 - 32);
          if ( *(_BYTE *)v16 == 61 )
          {
            v17 = sub_BD4070(*(v16 - 4), (__int64)v15);
            if ( v17 )
            {
              if ( *v17 > 0x1Cu )
              {
                v48[0] = v17;
                v8 = *(_BYTE **)(a1 + 32);
                if ( v8 != *(_BYTE **)(a1 + 40) )
                {
                  if ( v8 )
                  {
                    *(_QWORD *)v8 = v17;
                    v8 = *(_BYTE **)(a1 + 32);
                  }
                  *(_QWORD *)(a1 + 32) = v8 + 8;
                  return;
                }
LABEL_228:
                sub_24454E0(a1 + 24, v8, v48);
                return;
              }
            }
          }
        }
      }
      return;
    }
    if ( v4 == 69 )
    {
      if ( sub_B491E0(a2) )
      {
        v48[0] = a2;
        v12 = *(_BYTE **)(a1 + 8);
        if ( v12 == *(_BYTE **)(a1 + 16) )
        {
          sub_2445670(a1, v12, v48);
        }
        else
        {
          if ( v12 )
          {
            *(_QWORD *)v12 = a2;
            v12 = *(_BYTE **)(a1 + 8);
          }
          v12 += 8;
          *(_QWORD *)(a1 + 8) = v12;
        }
        if ( *(_DWORD *)(a1 + 48) == 1 && sub_B491E0(a2) )
        {
          v13 = *(unsigned __int8 ***)(a2 - 32);
          if ( *(_BYTE *)v13 == 61 )
          {
            v14 = sub_BD4070(*(v13 - 4), (__int64)v12);
            if ( v14 )
            {
              if ( *v14 > 0x1Cu )
              {
                v48[0] = v14;
                v8 = *(_BYTE **)(a1 + 32);
                if ( v8 == *(_BYTE **)(a1 + 40) )
                  goto LABEL_228;
                if ( v8 )
                  *(_QWORD *)v8 = v14;
                *(_QWORD *)(a1 + 32) += 8LL;
              }
            }
          }
        }
      }
      return;
    }
LABEL_115:
    if ( sub_B491E0(a2) )
    {
      v48[0] = a2;
      v27 = *(_BYTE **)(a1 + 8);
      if ( v27 == *(_BYTE **)(a1 + 16) )
      {
        sub_2445670(a1, v27, v48);
      }
      else
      {
        if ( v27 )
        {
          *(_QWORD *)v27 = a2;
          v27 = *(_BYTE **)(a1 + 8);
        }
        v27 += 8;
        *(_QWORD *)(a1 + 8) = v27;
      }
      if ( *(_DWORD *)(a1 + 48) == 1 && sub_B491E0(a2) )
      {
        v28 = *(unsigned __int8 ***)(a2 - 32);
        if ( *(_BYTE *)v28 == 61 )
        {
          v29 = sub_BD4070(*(v28 - 4), (__int64)v27);
          if ( v29 )
          {
            if ( *v29 > 0x1Cu )
            {
              v48[0] = v29;
              v8 = *(_BYTE **)(a1 + 32);
              if ( v8 == *(_BYTE **)(a1 + 40) )
                goto LABEL_228;
              if ( v8 )
                *(_QWORD *)v8 = v29;
              *(_QWORD *)(a1 + 32) += 8LL;
            }
          }
        }
      }
    }
    return;
  }
  if ( v4 == 71 )
  {
    if ( sub_B491E0(a2) )
    {
      v48[0] = a2;
      v21 = *(_BYTE **)(a1 + 8);
      if ( v21 == *(_BYTE **)(a1 + 16) )
      {
        sub_2445670(a1, v21, v48);
      }
      else
      {
        if ( v21 )
        {
          *(_QWORD *)v21 = a2;
          v21 = *(_BYTE **)(a1 + 8);
        }
        v21 += 8;
        *(_QWORD *)(a1 + 8) = v21;
      }
      if ( *(_DWORD *)(a1 + 48) == 1 && sub_B491E0(a2) )
      {
        v22 = *(unsigned __int8 ***)(a2 - 32);
        if ( *(_BYTE *)v22 == 61 )
        {
          v23 = sub_BD4070(*(v22 - 4), (__int64)v21);
          if ( v23 )
          {
            if ( *v23 > 0x1Cu )
            {
              v48[0] = v23;
              v8 = *(_BYTE **)(a1 + 32);
              if ( v8 == *(_BYTE **)(a1 + 40) )
                goto LABEL_228;
              if ( v8 )
                *(_QWORD *)v8 = v23;
              *(_QWORD *)(a1 + 32) += 8LL;
            }
          }
        }
      }
    }
    return;
  }
  if ( v4 != 154 )
    goto LABEL_115;
  if ( sub_B491E0(a2) )
  {
    v48[0] = a2;
    v5 = *(_BYTE **)(a1 + 8);
    if ( v5 == *(_BYTE **)(a1 + 16) )
    {
      sub_2445670(a1, v5, v48);
    }
    else
    {
      if ( v5 )
      {
        *(_QWORD *)v5 = a2;
        v5 = *(_BYTE **)(a1 + 8);
      }
      v5 += 8;
      *(_QWORD *)(a1 + 8) = v5;
    }
    if ( *(_DWORD *)(a1 + 48) == 1 && sub_B491E0(a2) )
    {
      v6 = *(unsigned __int8 ***)(a2 - 32);
      if ( *(_BYTE *)v6 == 61 )
      {
        v7 = sub_BD4070(*(v6 - 4), (__int64)v5);
        if ( v7 )
        {
          if ( *v7 > 0x1Cu )
          {
            v48[0] = v7;
            v8 = *(_BYTE **)(a1 + 32);
            if ( v8 == *(_BYTE **)(a1 + 40) )
              goto LABEL_228;
            if ( v8 )
              *(_QWORD *)v8 = v7;
            *(_QWORD *)(a1 + 32) += 8LL;
          }
        }
      }
    }
  }
}
