// Function: sub_E327B0
// Address: 0xe327b0
//
void __fastcall sub_E327B0(__int64 a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 v2; // rcx
  char v3; // r13
  __int64 v4; // rcx
  char v5; // al
  char v6; // dl
  unsigned __int64 v7; // rax
  __int64 v8; // r14
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // r15
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // r13
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // r15
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rcx
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // r13
  unsigned __int64 v25; // rax
  __int64 v26; // r13
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // r13
  unsigned __int64 v29; // rsi
  char *v30; // rax
  char *v31; // r13
  char v32; // si
  char *v33; // [rsp+8h] [rbp-68h]
  int v34; // [rsp+1Ch] [rbp-54h] BYREF
  size_t v35; // [rsp+20h] [rbp-50h] BYREF
  void *v36; // [rsp+28h] [rbp-48h]
  char v37; // [rsp+30h] [rbp-40h]

  if ( *(_BYTE *)(a1 + 49) || (v1 = *(_QWORD *)(a1 + 8), v1 >= *(_QWORD *)a1) )
  {
    *(_BYTE *)(a1 + 49) = 1;
    return;
  }
  v2 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)(a1 + 8) = v1 + 1;
  if ( v2 < *(_QWORD *)(a1 + 24) )
  {
    *(_QWORD *)(a1 + 40) = v2 + 1;
    v3 = *(_BYTE *)(*(_QWORD *)(a1 + 32) + v2);
    if ( !(unsigned __int8)sub_E313F0(v3, &v34) )
    {
      switch ( v3 )
      {
        case 'A':
          sub_E31C60(a1, 1u, "[");
          sub_E327B0(a1);
          sub_E31C60(a1, 2u, "; ");
          sub_E323E0(a1);
          sub_E31C60(a1, 1u, "]");
          goto LABEL_17;
        case 'B':
          if ( *(_BYTE *)(a1 + 49)
            || (v23 = *(_QWORD *)(a1 + 40), v23 >= *(_QWORD *)(a1 + 24))
            || *(_BYTE *)(*(_QWORD *)(a1 + 32) + v23) != 95 )
          {
            v25 = sub_E31BC0(a1);
            if ( *(_BYTE *)(a1 + 49) )
              goto LABEL_16;
            v24 = *(_QWORD *)(a1 + 40);
          }
          else
          {
            v24 = v23 + 1;
            v25 = 0;
            *(_QWORD *)(a1 + 40) = v24;
          }
          if ( v25 >= v24 )
            goto LABEL_16;
          if ( *(_BYTE *)(a1 + 48) )
          {
            *(_QWORD *)(a1 + 40) = v25;
            sub_E327B0(a1);
            *(_QWORD *)(a1 + 40) = v24;
          }
          goto LABEL_17;
        case 'D':
          v19 = *(_QWORD *)(a1 + 16);
          sub_E31C60(a1, 4u, "dyn ");
          sub_E321A0(a1);
          if ( *(_BYTE *)(a1 + 49) )
            goto LABEL_15;
          v8 = 0;
          while ( 2 )
          {
            v20 = *(_QWORD *)(a1 + 40);
            v21 = *(_QWORD *)(a1 + 24);
            if ( v20 < v21 )
            {
              v22 = *(_QWORD *)(a1 + 32);
              if ( *(_BYTE *)(v22 + v20) == 69 )
              {
                *(_QWORD *)(a1 + 16) = v19;
                *(_QWORD *)(a1 + 40) = v20 + 1;
                if ( v21 > v20 + 1 && *(_BYTE *)(v22 + v20 + 1) == 76 )
                {
                  *(_QWORD *)(a1 + 40) = v20 + 2;
                  if ( v20 + 2 < v21 && *(_BYTE *)(v22 + v20 + 2) == 95 )
                  {
                    *(_QWORD *)(a1 + 40) = v20 + 3;
                  }
                  else
                  {
                    v28 = sub_E31BC0(a1);
                    if ( v28 )
                    {
                      sub_E31C60(a1, 3u, " + ");
                      sub_E31E00(a1, v28);
                    }
                  }
                  goto LABEL_17;
                }
LABEL_16:
                *(_BYTE *)(a1 + 49) = 1;
                goto LABEL_17;
              }
            }
            if ( v8 )
              sub_E31C60(a1, 3u, " + ");
            v5 = sub_E331B0(a1, 1, 1);
            v6 = v5;
            if ( *(_BYTE *)(a1 + 49) )
            {
              if ( !v5 )
              {
LABEL_15:
                *(_QWORD *)(a1 + 16) = v19;
                goto LABEL_16;
              }
            }
            else
            {
              while ( 1 )
              {
                v7 = *(_QWORD *)(a1 + 40);
                if ( v7 >= *(_QWORD *)(a1 + 24) || *(_BYTE *)(*(_QWORD *)(a1 + 32) + v7) != 112 )
                  break;
                *(_QWORD *)(a1 + 40) = v7 + 1;
                if ( v6 )
                  sub_E31C60(a1, 2u, ", ");
                else
                  sub_E31570(a1, 60);
                sub_E31F10((__int64)&v35, a1);
                sub_E31C60(a1, v35, v36);
                sub_E31C60(a1, 3u, " = ");
                sub_E327B0(a1);
                v6 = 1;
                if ( *(_BYTE *)(a1 + 49) )
                  goto LABEL_102;
              }
              if ( !v6 )
              {
LABEL_14:
                ++v8;
                if ( *(_BYTE *)(a1 + 49) )
                  goto LABEL_15;
                continue;
              }
            }
            break;
          }
LABEL_102:
          sub_E31C60(a1, 1u, ">");
          goto LABEL_14;
        case 'F':
          v12 = *(_QWORD *)(a1 + 16);
          sub_E321A0(a1);
          if ( *(_BYTE *)(a1 + 49) )
            goto LABEL_44;
          v13 = *(_QWORD *)(a1 + 40);
          if ( v13 >= *(_QWORD *)(a1 + 24) )
            goto LABEL_44;
          v14 = *(_QWORD *)(a1 + 32);
          if ( *(_BYTE *)(v14 + v13) != 85 )
            goto LABEL_38;
          *(_QWORD *)(a1 + 40) = v13 + 1;
          sub_E31C60(a1, 7u, "unsafe ");
          if ( !*(_BYTE *)(a1 + 49) )
          {
            v13 = *(_QWORD *)(a1 + 40);
            if ( v13 < *(_QWORD *)(a1 + 24) )
            {
              v14 = *(_QWORD *)(a1 + 32);
LABEL_38:
              if ( *(_BYTE *)(v14 + v13) == 75 )
              {
                *(_QWORD *)(a1 + 40) = v13 + 1;
                sub_E31C60(a1, 8u, "extern \"");
                if ( *(_BYTE *)(a1 + 49)
                  || (v15 = *(_QWORD *)(a1 + 40), v15 >= *(_QWORD *)(a1 + 24))
                  || *(_BYTE *)(*(_QWORD *)(a1 + 32) + v15) != 67 )
                {
                  sub_E31F10((__int64)&v35, a1);
                  v30 = (char *)v36;
                  if ( v37 )
                    *(_BYTE *)(a1 + 49) = 1;
                  v31 = v30;
                  v33 = &v30[v35];
                  while ( v33 != v31 )
                  {
                    v32 = *v31;
                    if ( *v31 == 95 )
                      v32 = 45;
                    ++v31;
                    sub_E31570(a1, v32);
                  }
                }
                else
                {
                  *(_QWORD *)(a1 + 40) = v15 + 1;
                  sub_E31C60(a1, 1u, "C");
                }
                sub_E31C60(a1, 2u, "\" ");
              }
            }
          }
LABEL_44:
          sub_E31C60(a1, 3u, "fn(");
          if ( *(_BYTE *)(a1 + 49) )
            goto LABEL_49;
          v16 = 0;
          while ( 1 )
          {
            v17 = *(_QWORD *)(a1 + 40);
            if ( v17 < *(_QWORD *)(a1 + 24) && *(_BYTE *)(*(_QWORD *)(a1 + 32) + v17) == 69 )
              break;
            if ( v16 )
              sub_E31C60(a1, 2u, ", ");
            ++v16;
            sub_E327B0(a1);
            if ( *(_BYTE *)(a1 + 49) )
              goto LABEL_49;
          }
          *(_QWORD *)(a1 + 40) = v17 + 1;
LABEL_49:
          sub_E31C60(a1, 1u, ")");
          if ( *(_BYTE *)(a1 + 49)
            || (v18 = *(_QWORD *)(a1 + 40), v18 >= *(_QWORD *)(a1 + 24))
            || *(_BYTE *)(*(_QWORD *)(a1 + 32) + v18) != 117 )
          {
            sub_E31C60(a1, 4u, " -> ");
            sub_E327B0(a1);
          }
          else
          {
            *(_QWORD *)(a1 + 40) = v18 + 1;
          }
          *(_QWORD *)(a1 + 16) = v12;
          goto LABEL_17;
        case 'O':
          sub_E31C60(a1, 5u, "*mut ");
          sub_E327B0(a1);
          goto LABEL_17;
        case 'P':
          sub_E31C60(a1, 7u, "*const ");
          sub_E327B0(a1);
          goto LABEL_17;
        case 'Q':
        case 'R':
          sub_E31570(a1, 38);
          if ( !*(_BYTE *)(a1 + 49) )
          {
            v9 = *(_QWORD *)(a1 + 40);
            v10 = *(_QWORD *)(a1 + 24);
            if ( v9 < v10 )
            {
              v11 = *(_QWORD *)(a1 + 32);
              if ( *(_BYTE *)(v11 + v9) == 76 )
              {
                *(_QWORD *)(a1 + 40) = v9 + 1;
                if ( v10 > v9 + 1 && *(_BYTE *)(v11 + v9 + 1) == 95 )
                {
                  *(_QWORD *)(a1 + 40) = v9 + 2;
                }
                else
                {
                  v29 = sub_E31BC0(a1);
                  if ( v29 )
                  {
                    sub_E31E00(a1, v29);
                    sub_E31570(a1, 32);
                  }
                }
              }
            }
          }
          if ( v3 == 81 )
            sub_E31C60(a1, 4u, "mut ");
          sub_E327B0(a1);
          goto LABEL_17;
        case 'S':
          sub_E31C60(a1, 1u, "[");
          sub_E327B0(a1);
          sub_E31C60(a1, 1u, "]");
          goto LABEL_17;
        case 'T':
          v26 = 0;
          sub_E31C60(a1, 1u, "(");
          if ( *(_BYTE *)(a1 + 49) )
            goto LABEL_77;
          while ( 1 )
          {
            v27 = *(_QWORD *)(a1 + 40);
            if ( v27 < *(_QWORD *)(a1 + 24) && *(_BYTE *)(*(_QWORD *)(a1 + 32) + v27) == 69 )
              break;
            if ( v26 )
              sub_E31C60(a1, 2u, ", ");
            ++v26;
            sub_E327B0(a1);
            if ( *(_BYTE *)(a1 + 49) )
              goto LABEL_75;
          }
          *(_QWORD *)(a1 + 40) = v27 + 1;
LABEL_75:
          if ( v26 == 1 )
            sub_E31C60(a1, 1u, ",");
LABEL_77:
          sub_E31C60(a1, 1u, ")");
          goto LABEL_17;
        default:
          goto LABEL_19;
      }
    }
LABEL_7:
    switch ( v34 )
    {
      case 0:
        sub_E31C60(a1, 4u, "bool");
        break;
      case 1:
        sub_E31C60(a1, 4u, "char");
        break;
      case 2:
        sub_E31C60(a1, 2u, "i8");
        break;
      case 3:
        sub_E31C60(a1, 3u, "i16");
        break;
      case 4:
        sub_E31C60(a1, 3u, "i32");
        break;
      case 5:
        sub_E31C60(a1, 3u, "i64");
        break;
      case 6:
        sub_E31C60(a1, 4u, "i128");
        break;
      case 7:
        sub_E31C60(a1, 5u, "isize");
        break;
      case 8:
        sub_E31C60(a1, 2u, "u8");
        break;
      case 9:
        sub_E31C60(a1, 3u, "u16");
        break;
      case 10:
        sub_E31C60(a1, 3u, "u32");
        break;
      case 11:
        sub_E31C60(a1, 3u, "u64");
        break;
      case 12:
        sub_E31C60(a1, 4u, "u128");
        break;
      case 13:
        sub_E31C60(a1, 5u, "usize");
        break;
      case 14:
        sub_E31C60(a1, 3u, "f32");
        break;
      case 15:
        sub_E31C60(a1, 3u, "f64");
        break;
      case 16:
        sub_E31C60(a1, 3u, "str");
        break;
      case 17:
        sub_E31C60(a1, 1u, "_");
        break;
      case 18:
        sub_E31C60(a1, 2u, "()");
        break;
      case 19:
        sub_E31C60(a1, 3u, "...");
        break;
      case 20:
        sub_E31C60(a1, 1u, &unk_3F6A4C5);
        break;
      default:
        goto LABEL_17;
    }
    goto LABEL_17;
  }
  *(_BYTE *)(a1 + 49) = 1;
  if ( (unsigned __int8)sub_E313F0(0, &v34) )
    goto LABEL_7;
LABEL_19:
  *(_QWORD *)(a1 + 40) = v4;
  sub_E331B0(a1, 1, 0);
LABEL_17:
  *(_QWORD *)(a1 + 8) = v1;
}
