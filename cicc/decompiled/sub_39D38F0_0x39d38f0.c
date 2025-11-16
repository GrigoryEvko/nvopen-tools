// Function: sub_39D38F0
// Address: 0x39d38f0
//
void __fastcall sub_39D38F0(__int64 *a1, __int64 a2, __int64 a3, _DWORD *a4, char a5, _BYTE *a6, unsigned int a7)
{
  unsigned int v10; // r8d
  unsigned __int8 *v11; // r14
  __int64 v13; // rsi
  _BYTE *v14; // rcx
  __int16 v15; // ax
  __int64 v16; // rax
  __int64 v17; // rcx
  char v18; // r8
  __int64 v19; // rdi
  __int64 (*v20)(); // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rsi
  unsigned int v26; // ecx
  __int64 *v27; // rdx
  __int64 v28; // r8
  __int64 v29; // r12
  __int64 v30; // rbx
  __int64 v31; // rdx
  __int64 v32; // rcx
  const char *v33; // rdi
  __int64 v34; // r8
  size_t v35; // rax
  unsigned int v36; // eax
  void *v37; // rdx
  char v38; // dl
  int v39; // ebx
  __int64 v40; // rdx
  int v41; // eax
  _BYTE *v42; // rax
  _BYTE *v43; // rax
  int v44; // edx
  int v45; // r9d
  __int64 v46; // [rsp+0h] [rbp-70h]
  unsigned int v48; // [rsp+8h] [rbp-68h]
  unsigned int v49; // [rsp+Ch] [rbp-64h]
  int v50; // [rsp+Ch] [rbp-64h]
  _QWORD v51[2]; // [rsp+10h] [rbp-60h] BYREF
  char *v52[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v53[8]; // [rsp+30h] [rbp-40h] BYREF

  v10 = a7;
  v11 = (unsigned __int8 *)(*(_QWORD *)(a2 + 32) + 40LL * (unsigned int)a3);
  v13 = *v11;
  v14 = a6;
  switch ( *v11 )
  {
    case 0u:
    case 2u:
    case 3u:
    case 4u:
    case 6u:
    case 7u:
    case 8u:
    case 9u:
    case 0xAu:
    case 0xBu:
    case 0xDu:
    case 0xEu:
    case 0xFu:
    case 0x10u:
    case 0x11u:
    case 0x12u:
      v49 = 0;
      if ( !*v11 && a5 && (*((_WORD *)v11 + 1) & 0xFF0) != 0 && (v11[3] & 0x10) == 0 )
      {
        v13 = (unsigned int)a3;
        v36 = sub_1E16AB0(*((_QWORD *)v11 + 2), a3, a3, (__int64)a6, a7, a6);
        v14 = a6;
        v10 = a7;
        v49 = v36;
      }
      goto LABEL_5;
    case 1u:
      v15 = **(_WORD **)(a2 + 16);
      switch ( v15 )
      {
        case 7:
          if ( (_DWORD)a3 != 2 )
            break;
LABEL_24:
          sub_1E318F0(*a1, (__int64)v11);
          sub_1E31810(*a1, *((_QWORD *)v11 + 3), (__int64)a4);
          return;
        case 8:
          if ( (_DWORD)a3 == 3 )
            goto LABEL_24;
          break;
        case 14:
          if ( (unsigned int)a3 > 1 && (a3 & 1) == 0 )
            goto LABEL_24;
          break;
        default:
          if ( (_DWORD)a3 == 3 && v15 == 10 )
            goto LABEL_24;
          break;
      }
      v49 = 0;
LABEL_5:
      v48 = v10;
      v46 = (__int64)v14;
      v16 = sub_1E15F70(a2);
      v17 = v46;
      v18 = v48;
      v19 = *(_QWORD *)(v16 + 8);
      v20 = *(__int64 (**)())(*(_QWORD *)v19 + 32LL);
      v21 = 0;
      if ( v20 != sub_16FF770 )
      {
        v21 = ((__int64 (__fastcall *)(__int64, __int64, __int64 (*)(), __int64, _QWORD))v20)(v19, v13, v20, v46, v48);
        v18 = v48;
        v17 = v46;
      }
      sub_1E32250((__int64)v11, *a1, a1[1], v17, v18, 0, a5, v49, (__int64)a4, v21);
      return;
    case 5u:
      sub_39D3860(a1, *((_DWORD *)v11 + 6));
      return;
    case 0xCu:
      v22 = a1[2];
      v23 = *((_QWORD *)v11 + 3);
      v24 = *(unsigned int *)(v22 + 24);
      if ( !(_DWORD)v24 )
        goto LABEL_25;
      v25 = *(_QWORD *)(v22 + 8);
      v26 = (v24 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v27 = (__int64 *)(v25 + 16LL * v26);
      v28 = *v27;
      if ( v23 == *v27 )
        goto LABEL_11;
      v44 = 1;
      while ( 2 )
      {
        if ( v28 == -4 )
        {
LABEL_25:
          v29 = *a1;
        }
        else
        {
          v45 = v44 + 1;
          v26 = (v24 - 1) & (v44 + v26);
          v27 = (__int64 *)(v25 + 16LL * v26);
          v28 = *v27;
          if ( v23 != *v27 )
          {
            v44 = v45;
            continue;
          }
LABEL_11:
          v29 = *a1;
          if ( v27 != (__int64 *)(v25 + 16 * v24) )
          {
            v30 = *((unsigned int *)v27 + 2);
            v33 = *(const char **)((*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)a4 + 56LL))(a4) + 8 * v30);
            v35 = 0;
            v51[0] = v33;
            if ( v33 )
              v35 = strlen(v33);
            v51[1] = v35;
            sub_16D2060(v52, v51, v31, v32, v34);
            sub_16E7EE0(v29, v52[0], (size_t)v52[1]);
            if ( (_QWORD *)v52[0] != v53 )
              j_j___libc_free_0((unsigned __int64)v52[0]);
            return;
          }
        }
        break;
      }
      v37 = *(void **)(v29 + 24);
      if ( *(_QWORD *)(v29 + 16) - (_QWORD)v37 <= 0xDu )
      {
        sub_16E7EE0(v29, "CustomRegMask(", 0xEu);
      }
      else
      {
        qmemcpy(v37, "CustomRegMask(", 14);
        *(_QWORD *)(v29 + 24) += 14LL;
      }
      v38 = 0;
      v39 = 0;
      v50 = a4[4];
      if ( v50 > 0 )
      {
        do
        {
          v41 = *(_DWORD *)(v23 + 4LL * (v39 >> 5));
          if ( _bittest(&v41, v39) )
          {
            if ( v38 )
            {
              v42 = *(_BYTE **)(v29 + 24);
              if ( (unsigned __int64)v42 >= *(_QWORD *)(v29 + 16) )
              {
                sub_16E7DE0(v29, 44);
              }
              else
              {
                *(_QWORD *)(v29 + 24) = v42 + 1;
                *v42 = 44;
              }
            }
            sub_1F4AA00((__int64 *)v52, v39, (__int64)a4, 0, 0);
            if ( !v53[0] )
              sub_4263D6(v52, (unsigned int)v39, v40);
            ((void (__fastcall *)(char **, __int64))v53[1])(v52, v29);
            if ( v53[0] )
              ((void (__fastcall *)(char **, char **, __int64))v53[0])(v52, v52, 3);
            v38 = 1;
          }
          ++v39;
        }
        while ( v50 != v39 );
      }
      v43 = *(_BYTE **)(v29 + 24);
      if ( (unsigned __int64)v43 >= *(_QWORD *)(v29 + 16) )
      {
        sub_16E7DE0(v29, 41);
      }
      else
      {
        *(_QWORD *)(v29 + 24) = v43 + 1;
        *v43 = 41;
      }
      return;
    default:
      return;
  }
}
