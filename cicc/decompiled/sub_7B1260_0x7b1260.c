// Function: sub_7B1260
// Address: 0x7b1260
//
_QWORD *sub_7B1260()
{
  _QWORD *result; // rax
  __int64 *v1; // r15
  bool v2; // al
  char *v3; // r14
  char v4; // cl
  int v5; // ebx
  int v6; // r12d
  char v7; // r13
  unsigned __int64 v8; // rax
  int v9; // r9d
  char *v10; // r12
  __int64 v11; // rdx
  unsigned __int8 v12; // si
  unsigned __int8 v13; // al
  char v14; // dl
  unsigned int v15; // eax
  __int64 v16; // rax
  char *v17; // rdx
  char v18; // al
  __int64 v19; // r14
  int v20; // eax
  __int64 *v21; // rax
  unsigned __int64 v22; // rdi
  const char *v23; // r12
  char *v24; // rbx
  int v25; // eax
  char *v26; // r13
  int v27; // edi
  int *v28; // rax
  int v29; // [rsp-84h] [rbp-84h]
  int v30; // [rsp-80h] [rbp-80h]
  char v31; // [rsp-7Ah] [rbp-7Ah]
  char v32; // [rsp-79h] [rbp-79h]
  char v33; // [rsp-78h] [rbp-78h]
  char v34; // [rsp-78h] [rbp-78h]
  __int64 v35; // [rsp-70h] [rbp-70h]
  unsigned __int64 v36; // [rsp-60h] [rbp-60h] BYREF
  char v37[88]; // [rsp-58h] [rbp-58h] BYREF

  if ( dword_4D03CF4 )
  {
    result = &qword_4F06438;
    if ( !qword_4F06438 || dword_4F17F9C )
      return result;
  }
  v30 = unk_4F064A8;
  if ( unk_4F0647C != dword_4D03CEC && unk_4D03CE8 && !qword_4F06438 )
  {
    if ( (unk_4F0647C <= (unsigned int)(dword_4D03CEC + 5) || dword_4F17F5C && unk_4D04938)
      && unk_4F0647C > dword_4D03CEC )
    {
      do
      {
        if ( dword_4D04930 )
          putc(10, qword_4D04928);
        ++dword_4D03CEC;
      }
      while ( unk_4F0647C != dword_4D03CEC );
    }
    else
    {
      sub_7AF280(32, 0);
    }
  }
  v1 = (__int64 *)unk_4F06458;
  v2 = v30 != 0 && qword_4F06440 == 0;
  if ( !unk_4F06458 )
  {
    if ( v2 )
    {
      v23 = (const char *)qword_4F06498;
      goto LABEL_124;
    }
    goto LABEL_12;
  }
  do
  {
    if ( *((_BYTE *)v1 + 20) )
      break;
    v1 = (__int64 *)*v1;
  }
  while ( v1 );
  if ( !v2
    || (v23 = (const char *)qword_4F06498, unk_4D04328) && (v23[strlen((const char *)qword_4F06498) + 1] != 2 || v1) )
  {
LABEL_12:
    v35 = qword_4F06438;
    if ( qword_4F06438 )
      v3 = *(char **)(qword_4F06438 + 56LL);
    else
      v3 = (char *)qword_4F06498;
    v32 = 0;
    v4 = *v3;
    v5 = 0;
    v6 = 0;
    v7 = 10;
    while ( 1 )
    {
      if ( !v1 || (char *)v1[1] != v3 )
        goto LABEL_17;
      v15 = *((_DWORD *)v1 + 4);
      if ( v15 == 2 )
        break;
      if ( v15 <= 2 )
      {
        if ( v15 )
        {
          fwrite("\\\n", 1u, 2u, qword_4D04928);
          ++dword_4D03CEC;
        }
        else
        {
          ++v3;
          fprintf(qword_4D04928, "??%c", (unsigned int)*((char *)v1 + 24));
        }
LABEL_61:
        v1 = (__int64 *)*v1;
        v4 = *v3;
        if ( v1 )
          goto LABEL_64;
        goto LABEL_68;
      }
      if ( v15 != 3 )
        sub_721090();
      v3 += 2;
      putc(0, qword_4D04928);
      v1 = (__int64 *)*v1;
      v4 = *v3;
      if ( v1 )
      {
LABEL_64:
        while ( !*((_BYTE *)v1 + 20) )
        {
          v1 = (__int64 *)*v1;
          if ( !v1 )
            goto LABEL_68;
        }
        v7 = 10;
      }
      else
      {
LABEL_68:
        v7 = 10;
LABEL_17:
        if ( v4 == 10 )
        {
          if ( (unsigned int)sub_7AF220((unsigned __int64)v3) )
          {
            v16 = sub_7AF1D0((unsigned __int64)v3);
            *(_BYTE *)(v16 + 48) |= 4u;
            v17 = *(char **)(v16 + 56);
            *(_QWORD *)(v16 + 24) = v35;
            if ( v17 == *(char **)(v16 + 64) )
            {
              v3 += *(_QWORD *)(v16 + 32);
            }
            else
            {
              v35 = v16;
              v3 = v17;
            }
            v4 = *v3;
            if ( *v3 )
            {
              v6 = 1;
            }
            else
            {
              v18 = v3[1];
              if ( v18 != 8 && v18 != 5 )
                v6 = 1;
            }
          }
          else
          {
            v7 = 10;
            ++v3;
            putc(10, qword_4D04928);
            unk_4D03CE8 = 1;
            ++dword_4D03CEC;
            v4 = *v3;
          }
        }
        else if ( v4 )
        {
          if ( v5 > 0 )
          {
            v9 = v4;
            --v5;
            v4 = 10;
            goto LABEL_31;
          }
          if ( !dword_4D0432C )
          {
            if ( v4 >= 0 )
              goto LABEL_40;
LABEL_70:
            if ( unk_4F064A8 | v30 )
            {
LABEL_40:
              v9 = v4;
              if ( v6 )
              {
                if ( !dword_4F04D98 )
                {
                  v12 = byte_4F04C80[v7 + 128];
                  if ( v12 != 1 )
                  {
                    v13 = byte_4F04C80[v4 + 128];
                    if ( v13 != 1 )
                    {
                      if ( v12 == v13 )
                        goto LABEL_55;
                      v14 = v7 & 0xDF;
                      if ( (v7 & 0xDF) == 0x45 && ((v4 - 43) & 0xFD) == 0 )
                        goto LABEL_55;
                      if ( ((v7 - 43) & 0xFD) == 0 && (v32 & 0xDF) == 0x45 )
                      {
                        if ( (unsigned int)(unsigned __int8)v4 - 48 <= 9 )
                          goto LABEL_55;
                        if ( v14 == 85 || v7 == 76 )
                        {
LABEL_51:
                          if ( v4 == 39 || v4 == 34 )
                            goto LABEL_55;
                        }
                      }
                      else
                      {
                        if ( v7 == 76 || v14 == 85 )
                          goto LABEL_51;
                        if ( v32 == 117 && v7 == 56 )
                        {
                          if ( v4 != 39 && v4 != 34 )
                          {
                            v32 = 56;
                            goto LABEL_31;
                          }
LABEL_55:
                          v31 = v4;
                          v29 = v4;
                          putc(32, qword_4D04928);
                          v32 = v7;
                          v9 = v29;
                          v4 = v31;
LABEL_31:
                          v33 = v4;
                          v10 = v3;
                          putc(v9, qword_4D04928);
                          v7 = v33;
                          goto LABEL_32;
                        }
                      }
                      if ( v7 == 34 && v13 == 2 )
                        goto LABEL_55;
                    }
                  }
                }
              }
              v32 = v7;
              goto LABEL_31;
            }
            v10 = v3;
            sub_722A20((unsigned __int8)v4, v37);
            putc(v37[0], qword_4D04928);
            v9 = v37[1];
            goto LABEL_72;
          }
          v5 = 0;
          if ( v4 >= 0 )
            goto LABEL_40;
          v34 = v4;
          v20 = sub_721AB0(v3, 0, unk_4F064A8 == 0);
          v4 = v34;
          v5 = v20 - 1;
          if ( v20 - 1 <= 0 )
            goto LABEL_70;
          if ( v30 )
          {
            v9 = v34;
            v4 = 10;
            goto LABEL_31;
          }
          v10 = &v3[v5];
          sub_722680((unsigned __int8 *)v3, &v36, 0, unk_4F064A8 == 0);
          if ( !unk_4F064A8 )
          {
            v24 = v37;
            v25 = sub_722A20(v36, v37);
            if ( v25 > 0 )
            {
              v26 = &v37[v25 - 1 + 1];
              do
              {
                v27 = *v24++;
                putc(v27, qword_4D04928);
              }
              while ( v26 != v24 );
            }
            v5 = 0;
            v7 = 10;
            goto LABEL_32;
          }
          v5 = 0;
          v9 = (char)v36;
          if ( v36 > 0xFF )
          {
            sprintf(v37, "%lx", v36);
            v22 = (unsigned __int64)v3;
            v3 = v10;
            sub_7B0EB0(v22, (__int64)dword_4F07508);
            sub_685190(0x689u, (__int64)v37);
            v9 = 63;
            v4 = 10;
            goto LABEL_31;
          }
LABEL_72:
          if ( (_BYTE)v9 )
          {
            v3 = v10;
            v4 = 10;
            goto LABEL_31;
          }
          v7 = 10;
LABEL_32:
          v3 = v10 + 1;
          unk_4D03CE8 = 0;
          v4 = v10[1];
          v6 = 0;
        }
        else
        {
          v8 = (unsigned __int8)v3[1];
          switch ( (_BYTE)v8 )
          {
            case 4:
              v4 = v3[2];
              v6 = 1;
              v3 += 2;
              break;
            case 5:
            case 8:
              goto LABEL_34;
            case 3:
              if ( qword_4F06438 == v35 && dword_4D03CF4 )
                goto LABEL_100;
              v19 = *(_QWORD *)(v35 + 16);
              if ( !v19 )
              {
                v19 = qword_4F06498;
                if ( unk_4F06478 )
                  v19 = unk_4F06470 + qword_4F06498;
              }
              v3 = (char *)(*(_QWORD *)(v35 + 32) + v19);
              if ( (*(_BYTE *)(v35 + 48) & 4) != 0 )
              {
                v4 = *v3;
                v6 = 1;
                v35 = *(_QWORD *)(v35 + 24);
              }
              else
              {
                v6 = 1;
                v21 = sub_7AF170(v35);
                v4 = *v3;
                v35 = (__int64)v21;
              }
              break;
            case 2:
              v3 += 2;
              v7 = 10;
              putc(10, qword_4D04928);
              unk_4D03CE8 = 1;
              ++dword_4D03CEC;
              v4 = *v3;
              break;
            case 1:
              goto LABEL_100;
            case 6:
              v7 = 0;
              v3 += 2;
              putc(0, qword_4D04928);
              v4 = *v3;
              break;
            default:
              if ( (unsigned __int8)v8 > 0xDu )
                goto LABEL_27;
              v11 = 9344;
              if ( _bittest64(&v11, v8) )
              {
LABEL_34:
                v4 = v3[2];
                v3 += 2;
              }
              else
              {
                switch ( (_BYTE)v8 )
                {
                  case 0xB:
                    v4 = v3[3];
                    v3 += 3;
                    break;
                  case 9:
                    goto LABEL_34;
                  case 0xC:
                    v4 = v3[2];
                    v3 += 2;
                    break;
                  default:
LABEL_27:
                    sub_721090();
                }
              }
              break;
          }
        }
      }
    }
    v3 += 2;
    putc(10, qword_4D04928);
    ++dword_4D03CEC;
    goto LABEL_61;
  }
LABEL_124:
  if ( fputs(v23, qword_4D04928) == -1 )
  {
    v28 = __errno_location();
    sub_6866A0(1513, *v28);
  }
  putc(10, qword_4D04928);
  ++dword_4D03CEC;
  unk_4D03CE8 = 1;
LABEL_100:
  dword_4F17F9C = 1;
  dword_4D03CF4 = 1;
  return &dword_4D03CF4;
}
