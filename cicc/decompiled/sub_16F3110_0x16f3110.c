// Function: sub_16F3110
// Address: 0x16f3110
//
unsigned __int64 __fastcall sub_16F3110(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 result; // rax
  __int64 v7; // rbx
  __int64 v8; // rbp
  __int64 v9; // r12
  _BYTE *v11; // rax
  __int64 v12; // rdx
  char *v13; // r14
  __int64 v14; // rbx
  char *i; // r13
  _BYTE *v16; // rax
  _BYTE *v17; // rax
  _BYTE *v18; // rax
  __int64 v19; // r13
  __int64 v20; // rbx
  __int64 v21; // rdi
  _BYTE *v22; // rax
  _DWORD *v23; // rdx
  __int64 v24; // xmm0_8
  const char *v25; // r13
  size_t v26; // rax
  _QWORD *v27; // rcx
  size_t v28; // rdx
  char *v29; // rsi
  char v30; // si
  unsigned __int64 v31; // rdi
  char *v32; // rcx
  const char *v33; // r13
  unsigned int v34; // ecx
  __int64 v35; // rsi
  char *v36; // [rsp-48h] [rbp-48h] BYREF
  char *v37; // [rsp-40h] [rbp-40h]
  __int64 v38; // [rsp-38h] [rbp-38h]
  int v39; // [rsp-30h] [rbp-30h]
  __int64 v40; // [rsp-28h] [rbp-28h]
  __int64 v41; // [rsp-20h] [rbp-20h]
  __int64 v42; // [rsp-8h] [rbp-8h]

  v42 = v8;
  v41 = v9;
  v40 = v7;
  switch ( *(_BYTE *)a1 )
  {
    case 0:
      v23 = *(_DWORD **)(a2 + 24);
      result = *(_QWORD *)(a2 + 16) - (_QWORD)v23;
      if ( result <= 3 )
      {
        v28 = 4;
        v29 = "null";
        goto LABEL_26;
      }
      *v23 = 1819047278;
      *(_QWORD *)(a2 + 24) += 4LL;
      break;
    case 1:
      v25 = "true";
      if ( !*(_BYTE *)(a1 + 8) )
        v25 = "false";
      v26 = strlen(v25);
      v27 = *(_QWORD **)(a2 + 24);
      v28 = v26;
      result = *(_QWORD *)(a2 + 16) - (_QWORD)v27;
      if ( v28 <= result )
      {
        if ( (unsigned int)v28 >= 8 )
        {
          v31 = (unsigned __int64)(v27 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          *v27 = *(_QWORD *)v25;
          *(_QWORD *)((char *)v27 + (unsigned int)v28 - 8) = *(_QWORD *)&v25[(unsigned int)v28 - 8];
          v32 = (char *)v27 - v31;
          v33 = (const char *)(v25 - v32);
          result = ((_DWORD)v28 + (_DWORD)v32) & 0xFFFFFFF8;
          if ( (unsigned int)result >= 8 )
          {
            result = ((_DWORD)v28 + (_DWORD)v32) & 0xFFFFFFF8;
            v34 = 0;
            do
            {
              v35 = v34;
              v34 += 8;
              *(_QWORD *)(v31 + v35) = *(_QWORD *)&v33[v35];
            }
            while ( v34 < (unsigned int)result );
          }
        }
        else if ( (v28 & 4) != 0 )
        {
          *(_DWORD *)v27 = *(_DWORD *)v25;
          result = (unsigned int)v28;
          *(_DWORD *)((char *)v27 + (unsigned int)v28 - 4) = *(_DWORD *)&v25[(unsigned int)v28 - 4];
        }
        else if ( (_DWORD)v28 )
        {
          result = *(unsigned __int8 *)v25;
          *(_BYTE *)v27 = result;
          if ( (v28 & 2) != 0 )
          {
            result = (unsigned int)v28;
            *(_WORD *)((char *)v27 + (unsigned int)v28 - 2) = *(_WORD *)&v25[(unsigned int)v28 - 2];
          }
        }
        *(_QWORD *)(a2 + 24) += v28;
      }
      else
      {
        v29 = (char *)v25;
LABEL_26:
        result = sub_16E7EE0(a2, v29, v28);
      }
      break;
    case 2:
      v24 = *(_QWORD *)(a1 + 8);
      v37 = "%.*g";
      v39 = 17;
      v38 = v24;
      v36 = (char *)&unk_49EF6A8;
      result = sub_16E8450(a2, (__int64)&v36, a3, a4, a5, a6);
      break;
    case 3:
      result = sub_16E7AB0(a2, *(_QWORD *)(a1 + 8));
      break;
    case 4:
    case 5:
      result = (unsigned __int64)sub_16F1A20(a2, *(unsigned __int8 **)(a1 + 8), *(_QWORD *)(a1 + 16));
      break;
    case 6:
      v11 = *(_BYTE **)(a2 + 24);
      if ( (unsigned __int64)v11 >= *(_QWORD *)(a2 + 16) )
      {
        sub_16E7DE0(a2, 123);
      }
      else
      {
        v12 = (__int64)(v11 + 1);
        *(_QWORD *)(a2 + 24) = v11 + 1;
        *v11 = 123;
      }
      sub_16F2F00(&v36, a1 + 8, v12, a4, a5);
      v13 = v37;
      if ( v36 != v37 )
      {
        v14 = *(_QWORD *)v36;
        for ( i = v36 + 8; ; i += 8 )
        {
          sub_16F1A20(a2, *(unsigned __int8 **)(v14 + 8), *(_QWORD *)(v14 + 16));
          v16 = *(_BYTE **)(a2 + 24);
          if ( (unsigned __int64)v16 >= *(_QWORD *)(a2 + 16) )
          {
            sub_16E7DE0(a2, 58);
          }
          else
          {
            *(_QWORD *)(a2 + 24) = v16 + 1;
            *v16 = 58;
          }
          sub_16F3110(v14 + 24, a2);
          if ( v13 == i )
            break;
          v14 = *(_QWORD *)i;
          v17 = *(_BYTE **)(a2 + 24);
          if ( (unsigned __int64)v17 >= *(_QWORD *)(a2 + 16) )
          {
            sub_16E7DE0(a2, 44);
          }
          else
          {
            *(_QWORD *)(a2 + 24) = v17 + 1;
            *v17 = 44;
          }
        }
        v13 = v36;
      }
      if ( v13 )
        j_j___libc_free_0(v13, v38 - (_QWORD)v13);
      result = *(_QWORD *)(a2 + 24);
      v30 = 125;
      if ( result >= *(_QWORD *)(a2 + 16) )
        goto LABEL_37;
      *(_QWORD *)(a2 + 24) = result + 1;
      *(_BYTE *)result = 125;
      break;
    case 7:
      v18 = *(_BYTE **)(a2 + 24);
      if ( (unsigned __int64)v18 >= *(_QWORD *)(a2 + 16) )
      {
        sub_16E7DE0(a2, 91);
      }
      else
      {
        *(_QWORD *)(a2 + 24) = v18 + 1;
        *v18 = 91;
      }
      v19 = *(_QWORD *)(a1 + 8);
      v20 = *(_QWORD *)(a1 + 16);
      if ( v19 != v20 )
      {
        while ( 1 )
        {
          v21 = v19;
          v19 += 40;
          sub_16F3110(v21, a2);
          if ( v20 == v19 )
            break;
          v22 = *(_BYTE **)(a2 + 24);
          if ( (unsigned __int64)v22 >= *(_QWORD *)(a2 + 16) )
          {
            sub_16E7DE0(a2, 44);
          }
          else
          {
            *(_QWORD *)(a2 + 24) = v22 + 1;
            *v22 = 44;
          }
        }
      }
      result = *(_QWORD *)(a2 + 24);
      if ( result >= *(_QWORD *)(a2 + 16) )
      {
        v30 = 93;
LABEL_37:
        result = sub_16E7DE0(a2, v30);
      }
      else
      {
        *(_QWORD *)(a2 + 24) = result + 1;
        *(_BYTE *)result = 93;
      }
      break;
    default:
      return result;
  }
  return result;
}
