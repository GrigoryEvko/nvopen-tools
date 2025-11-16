// Function: sub_B50F30
// Address: 0xb50f30
//
__int64 __fastcall sub_B50F30(int a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v4; // r13
  int v5; // esi
  int v7; // r8d
  __int64 v8; // r14
  unsigned __int8 v9; // bl
  unsigned int v10; // eax
  int v11; // r8d
  int v12; // r10d
  char v13; // dl
  unsigned int v14; // esi
  int v15; // r9d
  int v16; // eax
  unsigned __int8 v17; // dl
  unsigned __int8 v18; // dl
  unsigned __int8 v19; // dl
  __int64 v20; // rdi
  bool v21; // zf
  char v22; // bl
  __int64 v23; // r11
  char v24; // dl
  __int64 *v25; // rdx
  __int64 v26; // rax
  char v27; // dl
  __int64 v28; // rax
  char v29; // dl
  unsigned int v30; // ecx
  unsigned int v31; // [rsp+0h] [rbp-60h]
  char v32; // [rsp+7h] [rbp-59h]
  unsigned int v33; // [rsp+8h] [rbp-58h]
  unsigned int v34; // [rsp+Ch] [rbp-54h]
  __int64 v35; // [rsp+20h] [rbp-40h]
  char v36; // [rsp+28h] [rbp-38h]

  v4 = a2;
  v5 = *(unsigned __int8 *)(a2 + 8);
  LOBYTE(v3) = (_BYTE)v5 != 13 && (_BYTE)v5 != 7;
  if ( (_BYTE)v3 )
  {
    v7 = *(unsigned __int8 *)(a3 + 8);
    v8 = a3;
    v9 = v5;
    if ( (_BYTE)v7 == 13 || (_BYTE)v7 == 7 || (unsigned __int8)(v7 - 15) <= 1u || (unsigned __int8)(v5 - 15) <= 1u )
    {
      return 0;
    }
    else
    {
      v33 = v7 - 17;
      v32 = *(_BYTE *)(a3 + 8);
      v34 = v5 - 17;
      v31 = sub_BCB060(v4);
      v10 = sub_BCB060(v8);
      v13 = v32;
      v14 = v10;
      if ( v34 > 1 )
      {
        v12 = 0;
        v15 = 0;
      }
      else
      {
        v15 = *(_DWORD *)(v4 + 32);
        LOBYTE(v12) = v9 == 18;
      }
      if ( v33 > 1 )
      {
        v11 = 0;
        v16 = 0;
      }
      else
      {
        v16 = *(_DWORD *)(v8 + 32);
        LOBYTE(v11) = v32 == 18;
      }
      switch ( a1 )
      {
        case '&':
          if ( v34 <= 1 )
            v9 = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
          v3 = 0;
          if ( v9 == 12 )
          {
            if ( v33 <= 1 )
              v8 = **(_QWORD **)(v8 + 16);
            LOBYTE(v3) = v15 == v16 && *(_BYTE *)(v8 + 8) == 12;
            if ( (_BYTE)v3 )
            {
              LOBYTE(v3) = v31 > v14;
              LOBYTE(v16) = (_BYTE)v11 == (unsigned __int8)v12;
              v3 &= v16;
            }
          }
          break;
        case '\'':
        case '(':
          if ( v34 <= 1 )
            v9 = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
          v3 = 0;
          if ( v9 == 12 )
          {
            if ( v33 <= 1 )
              v8 = **(_QWORD **)(v8 + 16);
            LOBYTE(v3) = v15 == v16 && *(_BYTE *)(v8 + 8) == 12;
            if ( (_BYTE)v3 )
            {
              LOBYTE(v3) = (_BYTE)v12 == (unsigned __int8)v11;
              LOBYTE(v16) = v31 < v14;
              v3 &= v16;
            }
          }
          break;
        case ')':
        case '*':
          if ( v34 <= 1 )
            v9 = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
          if ( v9 <= 3u || v9 == 5 || (v3 = 0, (v9 & 0xFD) == 4) )
          {
            if ( v33 <= 1 )
              v8 = **(_QWORD **)(v8 + 16);
            LOBYTE(v3) = v15 == v16 && *(_BYTE *)(v8 + 8) == 12;
            if ( (_BYTE)v3 )
              goto LABEL_24;
          }
          break;
        case '+':
        case ',':
          if ( v34 <= 1 )
            v9 = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
          v3 = 0;
          if ( v9 == 12 )
          {
            if ( v33 <= 1 )
              v8 = **(_QWORD **)(v8 + 16);
            v17 = *(_BYTE *)(v8 + 8);
            if ( v17 <= 3u )
              goto LABEL_23;
            if ( v17 == 5 )
              goto LABEL_23;
            v3 = 0;
            if ( (v17 & 0xFD) == 4 )
              goto LABEL_23;
          }
          break;
        case '-':
          if ( v34 <= 1 )
            v9 = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
          if ( v9 <= 3u || v9 == 5 || (v3 = 0, (v9 & 0xFD) == 4) )
          {
            if ( v33 <= 1 )
              v8 = **(_QWORD **)(v8 + 16);
            v19 = *(_BYTE *)(v8 + 8);
            if ( v19 <= 3u || v19 == 5 || (v3 = 0, (v19 & 0xFD) == 4) )
            {
              v3 = 0;
              if ( v15 == v16 )
              {
                LOBYTE(v3) = v31 > v14;
                LOBYTE(v16) = (_BYTE)v12 == (unsigned __int8)v11;
                v3 &= v16;
              }
            }
          }
          break;
        case '.':
          if ( v34 <= 1 )
            v9 = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
          if ( v9 <= 3u || v9 == 5 || (v3 = 0, (v9 & 0xFD) == 4) )
          {
            if ( v33 <= 1 )
              v8 = **(_QWORD **)(v8 + 16);
            v18 = *(_BYTE *)(v8 + 8);
            if ( v18 <= 3u || v18 == 5 || (v3 = 0, (v18 & 0xFD) == 4) )
            {
              v3 = 0;
              if ( v15 == v16 )
              {
                LOBYTE(v3) = v31 < v14;
                LOBYTE(v16) = (_BYTE)v12 == (unsigned __int8)v11;
                v3 &= v16;
              }
            }
          }
          break;
        case '/':
          v3 = 0;
          if ( v15 == v16 && (_BYTE)v12 == (_BYTE)v11 )
          {
            if ( v34 <= 1 )
              v4 = **(_QWORD **)(v4 + 16);
            v3 = 0;
            if ( *(_BYTE *)(v4 + 8) == 14 )
            {
              if ( v33 <= 1 )
                v8 = **(_QWORD **)(v8 + 16);
              LOBYTE(v3) = *(_BYTE *)(v8 + 8) == 12;
            }
          }
          break;
        case '0':
          v3 = 0;
          if ( v15 == v16 && (_BYTE)v12 == (_BYTE)v11 )
          {
            if ( v34 <= 1 )
              v4 = **(_QWORD **)(v4 + 16);
            v3 = 0;
            if ( *(_BYTE *)(v4 + 8) == 12 )
            {
              if ( v33 <= 1 )
                v8 = **(_QWORD **)(v8 + 16);
              LOBYTE(v3) = *(_BYTE *)(v8 + 8) == 14;
            }
          }
          break;
        case '1':
          v20 = v4;
          if ( v34 <= 1 )
          {
            v20 = **(_QWORD **)(v4 + 16);
            v9 = *(_BYTE *)(v20 + 8);
          }
          v21 = v9 == 14;
          v22 = 0;
          if ( !v21 )
          {
            v20 = 0;
            v22 = v3;
          }
          v23 = v8;
          if ( v33 <= 1 )
          {
            v23 = **(_QWORD **)(v8 + 16);
            v13 = *(_BYTE *)(v23 + 8);
          }
          v21 = v13 == 14;
          v24 = 0;
          if ( !v21 )
          {
            v24 = v3;
            v23 = 0;
          }
          if ( v24 != v22 )
            return 0;
          if ( v20 )
          {
            if ( *(_DWORD *)(v23 + 8) >> 8 != *(_DWORD *)(v20 + 8) >> 8 )
              return 0;
            if ( v34 <= 1 && v33 <= 1 )
            {
LABEL_23:
              v3 = 0;
              if ( v15 == v16 )
                goto LABEL_24;
            }
            else if ( v34 > 1 )
            {
              if ( v33 <= 1 )
              {
                v3 = 0;
                if ( v16 == 1 )
                  v3 = v11 ^ 1;
              }
            }
            else
            {
              v3 = 0;
              if ( v15 == 1 )
                v3 = v12 ^ 1;
            }
          }
          else
          {
            v26 = sub_BCAE30(v8);
            v36 = v27;
            v35 = v26;
            v28 = sub_BCAE30(v4);
            v30 = 0;
            if ( v28 == v35 )
              LOBYTE(v30) = v29 == v36;
            v3 = v30;
          }
          break;
        case '2':
          if ( v34 <= 1 )
          {
            v25 = *(__int64 **)(v4 + 16);
            v4 = *v25;
            v9 = *(_BYTE *)(*v25 + 8);
          }
          v3 = 0;
          if ( v9 == 14 )
          {
            if ( v33 <= 1 )
              v8 = **(_QWORD **)(v8 + 16);
            v3 = 0;
            if ( *(_BYTE *)(v8 + 8) == 14 && *(_DWORD *)(v8 + 8) >> 8 != *(_DWORD *)(v4 + 8) >> 8 && v15 == v16 )
LABEL_24:
              LOBYTE(v3) = (_BYTE)v12 == (unsigned __int8)v11;
          }
          break;
        default:
          v3 = 0;
          break;
      }
    }
  }
  return v3;
}
