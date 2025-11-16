// Function: sub_18C6ED0
// Address: 0x18c6ed0
//
char __fastcall sub_18C6ED0(__int64 a1)
{
  const char *v1; // rax
  __int64 v2; // r12
  unsigned __int64 v3; // rax
  size_t v4; // r13
  unsigned __int64 v5; // rbx
  void *v6; // r15
  unsigned __int64 v7; // r14
  char *v8; // rbx
  __int64 v9; // rdx
  bool v10; // di
  char v11; // al
  char v12; // dl
  char v13; // r15
  bool v14; // r9
  char v15; // r8
  char v16; // r10
  bool v17; // si
  char v18; // r9
  bool v19; // r10
  bool v20; // r9
  char v21; // al
  char v22; // r10
  char v23; // r9
  char v24; // di
  char v25; // r9
  char v27; // [rsp+Ch] [rbp-74h]
  __int64 v28; // [rsp+20h] [rbp-60h]
  char v30; // [rsp+3Fh] [rbp-41h] BYREF
  void *s1; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 v32; // [rsp+48h] [rbp-38h]

  LOBYTE(v1) = qword_4FADF68;
  v2 = qword_4FADF60;
  v28 = qword_4FADF68;
  if ( qword_4FADF68 != qword_4FADF60 )
  {
    while ( 1 )
    {
      s1 = *(void **)v2;
      v3 = *(_QWORD *)(v2 + 8);
      v30 = 58;
      v32 = v3;
      v4 = sub_16D20C0((__int64 *)&s1, &v30, 1u, 0);
      if ( v4 == -1 )
      {
        v6 = s1;
        v4 = v32;
        v7 = 0;
        v8 = 0;
      }
      else
      {
        v5 = v4 + 1;
        v6 = s1;
        if ( v4 + 1 > v32 )
          v5 = v32;
        v7 = v32 - v5;
        v8 = (char *)s1 + v5;
        if ( v4 && v4 > v32 )
          v4 = v32;
      }
      v1 = sub_1649960(a1);
      if ( v4 != v9 )
        goto LABEL_3;
      if ( v4 )
      {
        LODWORD(v1) = memcmp(v6, v1, v4);
        if ( (_DWORD)v1 )
          goto LABEL_3;
      }
      if ( v7 == 12 )
        break;
      if ( v7 == 7 )
      {
        if ( *(_DWORD *)v8 != 1818850658 || *((_WORD *)v8 + 2) != 26996 || v8[6] != 110 )
        {
          if ( *(_DWORD *)v8 == 1936615789 && *((_WORD *)v8 + 2) == 31337 && v8[6] == 101 )
          {
            v27 = 17;
            v10 = 0;
            v12 = 1;
            v11 = 0;
            v13 = 1;
            v14 = 0;
          }
          else
          {
            v12 = 0;
            v10 = 0;
            v13 = 0;
            v11 = 1;
            v14 = 0;
          }
          goto LABEL_32;
        }
        v27 = 5;
        v10 = 0;
LABEL_107:
        v11 = 0;
        v12 = 1;
        v13 = 1;
        goto LABEL_20;
      }
      if ( v7 == 4 )
      {
        if ( *(_DWORD *)v8 == 1684828003 )
        {
          v27 = 7;
          v10 = 0;
          goto LABEL_107;
        }
        goto LABEL_27;
      }
      v10 = v7 == 9;
      if ( v7 == 10 )
      {
        if ( *(_QWORD *)v8 == 0x65677265766E6F63LL && *((_WORD *)v8 + 4) == 29806 )
        {
          v27 = 8;
          goto LABEL_107;
        }
        if ( *(_QWORD *)v8 == 0x6968656E696C6E69LL && *((_WORD *)v8 + 4) == 29806 )
        {
          v27 = 15;
          goto LABEL_107;
        }
LABEL_19:
        v11 = 1;
        v12 = 0;
        v13 = 0;
LABEL_20:
        if ( ((unsigned __int8)v11 & (v7 == 5)) != 0 )
        {
          v14 = v7 == 11;
          if ( *(_DWORD *)v8 == 1701536110 && v8[4] == 100 )
          {
            v27 = 18;
            v12 = v11 & (v7 == 5);
            v13 = 1;
            v15 = v7 == 15;
            v11 = 0;
LABEL_33:
            v16 = v15 & v11;
            v17 = v7 == 8;
            if ( ((unsigned __int8)v15 & (unsigned __int8)v11) != 0 )
              goto LABEL_67;
            goto LABEL_34;
          }
          goto LABEL_32;
        }
        goto LABEL_31;
      }
      if ( v7 != 9 )
        goto LABEL_27;
      if ( *(_QWORD *)v8 == 0x6C626174706D756ALL && v8[8] == 101 )
      {
        v27 = 16;
        v10 = 1;
        v11 = 0;
        v13 = 1;
        v12 = 1;
      }
      else
      {
        v11 = 1;
        v12 = 0;
        v10 = 1;
        v13 = 0;
      }
LABEL_31:
      v14 = v7 == 11;
      if ( ((unsigned __int8)v11 & v10) != 0 )
      {
        v15 = v7 == 15;
        if ( *(_QWORD *)v8 == 0x69746C6975626F6ELL && v8[8] == 110 )
        {
          v27 = 21;
          v12 = v11 & v10;
          v13 = 1;
          v17 = v7 == 8;
          v11 = 0;
        }
        else
        {
          v16 = v11 & v15;
          v17 = v7 == 8;
          if ( ((unsigned __int8)v11 & (unsigned __int8)v15) != 0 )
          {
LABEL_67:
            if ( *(_QWORD *)v8 == 0x63696C706D696F6ELL
              && *((_DWORD *)v8 + 2) == 1818653801
              && *((_WORD *)v8 + 6) == 24943
              && v8[14] == 116 )
            {
              v27 = 25;
              v12 = v16;
              v13 = 1;
              v11 = 0;
              goto LABEL_36;
            }
            goto LABEL_35;
          }
        }
        goto LABEL_34;
      }
LABEL_32:
      v15 = v7 == 15;
      if ( ((unsigned __int8)v11 & v14) == 0 )
        goto LABEL_33;
      v17 = v7 == 8;
      if ( *(_QWORD *)v8 == 0x63696C7075646F6ELL && *((_WORD *)v8 + 4) == 29793 && v8[10] == 101 )
      {
        v27 = 24;
        v12 = v11 & v14;
        v11 = 0;
        v13 = 1;
LABEL_35:
        v18 = v11 & v14;
        if ( v18 )
        {
          if ( *(_QWORD *)v8 == 0x62797A616C6E6F6ELL && *((_WORD *)v8 + 4) == 28265 && v8[10] == 100 )
          {
            v27 = 31;
            v12 = v18;
            v13 = 1;
            v19 = v7 == 10;
            v11 = 0;
LABEL_38:
            if ( (v19 & (unsigned __int8)v11) != 0 )
            {
              if ( *(_QWORD *)v8 == 0x6568635F66636F6ELL && *((_WORD *)v8 + 4) == 27491 )
              {
                v27 = 23;
                v12 = v19 & v11;
                v13 = 1;
                v20 = v7 == 13;
                v11 = 0;
                goto LABEL_41;
              }
              goto LABEL_40;
            }
            goto LABEL_39;
          }
          goto LABEL_37;
        }
        goto LABEL_36;
      }
LABEL_34:
      if ( (v17 & (unsigned __int8)v11) == 0 )
        goto LABEL_35;
      if ( *(_QWORD *)v8 == 0x656E696C6E696F6ELL )
      {
        v27 = 26;
        v12 = v17 & v11;
        v11 = 0;
        v13 = 1;
LABEL_37:
        v19 = v7 == 10;
        if ( (v17 & (unsigned __int8)v11) == 0 )
          goto LABEL_38;
        if ( *(_QWORD *)v8 == 0x6E72757465726F6ELL )
        {
          v27 = 29;
          v12 = v17 & v11;
          v11 = 0;
          v13 = 1;
          goto LABEL_40;
        }
        goto LABEL_39;
      }
LABEL_36:
      if ( (v10 & (unsigned __int8)v11) == 0 )
        goto LABEL_37;
      v19 = v7 == 10;
      if ( *(_QWORD *)v8 != 0x6E6F7A6465726F6ELL || v8[8] != 101 )
        goto LABEL_38;
      v27 = 28;
      v12 = v10 & v11;
      v11 = 0;
      v13 = 1;
LABEL_39:
      if ( (v10 & (unsigned __int8)v11) != 0 )
      {
        v20 = v7 == 13;
        if ( *(_QWORD *)v8 == 0x7372756365726F6ELL && v8[8] == 101 )
        {
          v27 = 27;
          goto LABEL_89;
        }
        goto LABEL_41;
      }
LABEL_40:
      v20 = v7 == 13;
      if ( (v17 & (unsigned __int8)v11) != 0 )
      {
        if ( *(_QWORD *)v8 == 0x646E69776E756F6ELL )
        {
          v27 = 30;
          goto LABEL_89;
        }
LABEL_42:
        if ( !v12 )
        {
          if ( v7 != 7 )
            goto LABEL_44;
          if ( *(_DWORD *)v8 == 1853124719 && *((_WORD *)v8 + 2) == 28271 && v8[6] == 101 )
          {
            v27 = 35;
          }
          else
          {
            if ( *(_DWORD *)v8 != 1937010799 || *((_WORD *)v8 + 2) != 31337 || v8[6] != 101 )
              goto LABEL_45;
            v27 = 34;
          }
        }
        goto LABEL_89;
      }
LABEL_41:
      if ( (v20 & (unsigned __int8)v11) == 0 )
        goto LABEL_42;
      if ( *(_QWORD *)v8 != 0x7566726F6674706FLL || *((_DWORD *)v8 + 2) != 1852406394 || v8[12] != 103 )
      {
LABEL_44:
        v12 = v13;
        if ( v17 )
        {
          if ( *(_QWORD *)v8 == 0x656E6F6E64616572LL )
          {
            v27 = 36;
LABEL_148:
            v12 = v17;
            v21 = 0;
LABEL_92:
            v23 = v10 & v21;
            if ( (v10 & (unsigned __int8)v21) != 0 )
            {
LABEL_93:
              if ( *(_QWORD *)v8 == 0x6361747365666173LL && v8[8] == 107 )
              {
                v27 = 41;
                v12 = v23;
                v21 = 0;
                goto LABEL_50;
              }
              goto LABEL_49;
            }
            goto LABEL_48;
          }
          if ( *(_QWORD *)v8 == 0x796C6E6F64616572LL )
          {
            v27 = 37;
            goto LABEL_148;
          }
          v12 = v13;
          v21 = v13 ^ 1;
          v25 = (v13 ^ 1) & v20;
          goto LABEL_91;
        }
LABEL_45:
        v21 = v12 ^ 1;
        v22 = (v12 ^ 1) & v19;
        if ( v22 )
        {
          if ( *(_QWORD *)v8 == 0x6E6F6D656D677261LL && *((_WORD *)v8 + 4) == 31084 )
          {
            v27 = 4;
            v12 = v22;
            v21 = 0;
          }
          else
          {
            v23 = v10;
            v21 = v22;
            v12 = 0;
            if ( v10 )
              goto LABEL_93;
          }
          goto LABEL_48;
        }
        goto LABEL_90;
      }
      v27 = 33;
LABEL_89:
      v12 = 1;
      v21 = 0;
LABEL_90:
      v25 = v21 & v20;
LABEL_91:
      if ( !v25 )
        goto LABEL_92;
      if ( *(_QWORD *)v8 == 0x5F736E7275746572LL && *((_DWORD *)v8 + 2) == 1667856244 && v8[12] == 101 )
      {
        v27 = 39;
        v12 = v25;
        v21 = 0;
        goto LABEL_49;
      }
LABEL_48:
      if ( ((unsigned __int8)v15 & (unsigned __int8)v21) != 0 )
      {
        if ( *(_QWORD *)v8 == 0x6163776F64616873LL
          && *((_DWORD *)v8 + 2) == 1953721452
          && *((_WORD *)v8 + 6) == 25441
          && v8[14] == 107 )
        {
          v27 = 46;
          goto LABEL_76;
        }
        goto LABEL_50;
      }
LABEL_49:
      if ( ((unsigned __int8)v21 & (v7 == 16)) != 0 )
      {
        if ( !(*(_QWORD *)v8 ^ 0x657A6974696E6173LL | *((_QWORD *)v8 + 1) ^ 0x737365726464615FLL) )
        {
          v27 = 42;
LABEL_76:
          v15 = 1;
          v12 = 0;
          goto LABEL_77;
        }
        goto LABEL_51;
      }
LABEL_50:
      if ( ((unsigned __int8)v21 & (v7 == 18)) != 0 )
      {
        if ( *(_QWORD *)v8 ^ 0x657A6974696E6173LL | *((_QWORD *)v8 + 1) ^ 0x657264646177685FLL
          || *((_WORD *)v8 + 8) != 29555 )
        {
          v15 = v12;
          v12 = v21;
          goto LABEL_54;
        }
        v27 = 43;
        v15 = v21 & (v7 == 18);
        v12 = 0;
LABEL_77:
        LOBYTE(v1) = v12 & (v7 == 6);
LABEL_78:
        if ( (_BYTE)v1 )
        {
          if ( *(_DWORD *)v8 == 1919972211 && *((_WORD *)v8 + 2) == 29029 )
          {
            v27 = 50;
            v15 = (char)v1;
            v12 = 0;
            goto LABEL_58;
          }
          goto LABEL_57;
        }
        v24 = v12 & v10;
        goto LABEL_80;
      }
LABEL_51:
      if ( v12 )
        goto LABEL_76;
      if ( !v15 )
      {
        v12 = 1;
LABEL_54:
        LOBYTE(v1) = v12 & (v7 == 3);
        if ( (_BYTE)v1 )
        {
          if ( *(_WORD *)v8 == 29555 && v8[2] == 112 )
          {
            v27 = 49;
            v15 = v12 & (v7 == 3);
            v12 = 0;
          }
          else
          {
            v12 &= v7 == 3;
            if ( v10 )
              goto LABEL_81;
          }
          goto LABEL_57;
        }
        goto LABEL_77;
      }
      LOBYTE(v1) = 115;
      if ( *(_QWORD *)v8 == 0x657A6974696E6173LL
        && *((_DWORD *)v8 + 2) == 1835363679
        && *((_WORD *)v8 + 6) == 29295
        && v8[14] == 121 )
      {
        v27 = 44;
        v24 = 0;
      }
      else
      {
        LOBYTE(v1) = 115;
        if ( *(_QWORD *)v8 != 0x657A6974696E6173LL
          || *((_DWORD *)v8 + 2) != 1919448159
          || *((_WORD *)v8 + 6) != 24933
          || v8[14] != 100 )
        {
          v12 = v15;
          LOBYTE(v1) = v7 == 6;
          v15 = 0;
          goto LABEL_78;
        }
        v27 = 45;
        v24 = 0;
      }
LABEL_80:
      if ( v24 )
      {
LABEL_81:
        LOBYTE(v1) = 115;
        if ( *(_QWORD *)v8 == 0x6E6F727473707373LL && v8[8] == 103 )
        {
          v27 = 51;
          goto LABEL_60;
        }
        goto LABEL_58;
      }
LABEL_57:
      if ( ((unsigned __int8)v12 & v17) != 0 )
      {
        LOBYTE(v1) = 115;
        if ( *(_QWORD *)v8 == 0x7066746369727473LL )
        {
          v27 = 52;
          goto LABEL_60;
        }
LABEL_59:
        if ( v15 )
          goto LABEL_60;
        goto LABEL_3;
      }
LABEL_58:
      if ( ((v7 == 7) & (unsigned __int8)v12) == 0 )
        goto LABEL_59;
      if ( *(_DWORD *)v8 == 1635022709 && *((_WORD *)v8 + 2) == 27746 && v8[6] == 101 )
      {
        v27 = 56;
LABEL_60:
        LOBYTE(v1) = sub_1560180(a1 + 112, v27);
        if ( !(_BYTE)v1 )
          LOBYTE(v1) = sub_15E0D50(a1, -1, v27);
      }
LABEL_3:
      v2 += 32;
      if ( v28 == v2 )
        return (char)v1;
    }
    if ( *(_QWORD *)v8 == 0x6E69737961776C61LL && *((_DWORD *)v8 + 2) == 1701734764 )
    {
      v27 = 3;
      v10 = 0;
      goto LABEL_107;
    }
LABEL_27:
    v10 = 0;
    goto LABEL_19;
  }
  return (char)v1;
}
