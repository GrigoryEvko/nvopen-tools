// Function: sub_E31F10
// Address: 0xe31f10
//
__int64 __fastcall sub_E31F10(__int64 a1, __int64 a2)
{
  __int64 v2; // r10
  __int64 v3; // r9
  char v4; // r11
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // r12
  __int64 v7; // rdi
  char v8; // dl
  unsigned __int64 v9; // r8
  signed __int64 v10; // rcx
  unsigned __int64 v11; // rcx
  char *v12; // rbx
  signed __int64 v13; // rdx
  char *v14; // r12
  char *v15; // rsi
  char *v16; // r8

  v2 = a1;
  v3 = a2;
  v4 = *(_BYTE *)(a2 + 49);
  if ( v4 )
    goto LABEL_23;
  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(_QWORD *)(a2 + 24);
  if ( v5 >= v6 )
    goto LABEL_23;
  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_BYTE *)(v7 + v5);
  if ( v8 == 117 )
  {
    *(_QWORD *)(a2 + 40) = ++v5;
    if ( v6 <= v5 )
      goto LABEL_23;
    v8 = *(_BYTE *)(v7 + v5);
    v4 = 1;
  }
  if ( (unsigned __int8)(v8 - 48) <= 9u )
  {
    if ( v8 == 48 )
    {
      v9 = v5 + 1;
      *(_QWORD *)(a2 + 40) = v5 + 1;
      if ( v6 <= v5 + 1 )
      {
        if ( v6 >= v5 + 1 )
        {
          v13 = 0;
          v10 = 0;
          v12 = (char *)(v7 + v9);
          goto LABEL_46;
        }
LABEL_47:
        sub_222CF80("%s: __pos (which is %zu) > __size (which is %zu)", (char)"basic_string_view::substr");
      }
      v10 = 0;
      if ( *(_BYTE *)(v7 + v5 + 1) != 95 )
      {
LABEL_14:
        if ( v6 >= v9 )
        {
          v12 = (char *)(v7 + v9);
          v13 = v10;
          *(_QWORD *)(a2 + 40) = v10 + v9;
          if ( v10 >> 2 > 0 )
          {
            v14 = &v12[4 * (v10 >> 2)];
            v15 = (char *)(v7 + v9);
            while ( (unsigned __int8)sub_E313B0(*v15) )
            {
              if ( !(unsigned __int8)sub_E313B0(v15[1]) )
              {
                ++v15;
                break;
              }
              if ( !(unsigned __int8)sub_E313B0(v15[2]) )
              {
                v15 += 2;
                break;
              }
              if ( !(unsigned __int8)sub_E313B0(v15[3]) )
              {
                v15 += 3;
                break;
              }
              v15 += 4;
              if ( v14 == v15 )
              {
                v13 = v16 - v15;
                goto LABEL_32;
              }
            }
LABEL_22:
            if ( v16 != v15 )
              goto LABEL_23;
LABEL_36:
            *(_QWORD *)v2 = v10;
            *(_QWORD *)(v2 + 8) = v12;
            *(_BYTE *)(v2 + 16) = v4;
            return v2;
          }
LABEL_46:
          v15 = v12;
LABEL_32:
          if ( v13 != 2 )
          {
            if ( v13 != 3 )
            {
              if ( v13 != 1 )
                goto LABEL_36;
              goto LABEL_35;
            }
            if ( !(unsigned __int8)sub_E313B0(*v15) )
              goto LABEL_22;
            ++v15;
          }
          if ( !(unsigned __int8)sub_E313B0(*v15) )
            goto LABEL_22;
          ++v15;
LABEL_35:
          if ( (unsigned __int8)sub_E313B0(*v15) )
            goto LABEL_36;
          goto LABEL_22;
        }
        goto LABEL_47;
      }
      v9 = v5 + 2;
      *(_QWORD *)(a2 + 40) = v5 + 2;
    }
    else
    {
      v9 = *(_QWORD *)(a2 + 40);
      v10 = 0;
      while ( (unsigned __int8)(*(_BYTE *)(v7 + v9) - 48) <= 9u )
      {
        if ( !is_mul_ok(0xAu, v10) )
          goto LABEL_23;
        *(_QWORD *)(a2 + 40) = ++v9;
        v11 = 10 * v10;
        if ( v11 > ~(__int64)(*(char *)(v7 + v9 - 1) - 48) )
          goto LABEL_23;
        v10 = *(char *)(v7 + v9 - 1) - 48 + v11;
        if ( v6 <= v9 )
          goto LABEL_13;
      }
      v9 = *(_QWORD *)(a2 + 40);
      if ( v9 < v6 && *(_BYTE *)(v7 + v9) == 95 )
        *(_QWORD *)(a2 + 40) = ++v9;
    }
LABEL_13:
    if ( v6 - v9 >= v10 )
      goto LABEL_14;
  }
LABEL_23:
  *(_BYTE *)(v3 + 49) = 1;
  *(_QWORD *)(v2 + 16) = 0;
  *(_OWORD *)v2 = 0;
  return v2;
}
