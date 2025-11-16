// Function: sub_1421A90
// Address: 0x1421a90
//
__int64 __fastcall sub_1421A90(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  int v5; // ebx
  __int64 v6; // r12
  unsigned int v7; // r15d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned int v13; // r8d
  __int64 *v14; // rdx
  __int64 v15; // r10
  __int64 v16; // rax
  __int64 v17; // r9
  __int64 v18; // rdi
  unsigned int v19; // esi
  __int64 *v20; // rdx
  __int64 v21; // r8
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rdi
  unsigned int v25; // eax
  __int64 v26; // rax
  int v27; // edx
  __int64 v28; // rdx
  _QWORD *v29; // rax
  __int64 v30; // rsi
  unsigned __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rdi
  __int64 result; // rax
  __int64 v36; // rsi
  unsigned int v37; // ecx
  __int64 *v38; // rdx
  __int64 v39; // r9
  unsigned __int64 *v40; // r12
  unsigned __int64 *v41; // rbx
  __int64 v42; // rcx
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rdx
  __int64 v45; // rdx
  unsigned __int64 *v46; // rdi
  unsigned __int64 *v47; // rdi
  unsigned __int64 v48; // rdx
  int v49; // edx
  int v50; // r11d
  int v51; // edx
  int v52; // r10d
  __int64 v53; // rsi
  int v54; // edx
  int v55; // r8d
  __int64 v56; // [rsp+8h] [rbp-48h]
  __int64 v57; // [rsp+10h] [rbp-40h]
  __int64 v58; // [rsp+18h] [rbp-38h]

  v4 = sub_157EBA0(a2);
  if ( v4 )
  {
    v5 = sub_15F4D60(v4);
    v6 = sub_157EBA0(a2);
    if ( v5 )
    {
      v7 = 0;
      while ( 1 )
      {
        v8 = sub_15F4DF0(v6, v7);
        v9 = *(_QWORD *)(a1 + 8);
        v10 = v8;
        v11 = *(unsigned int *)(v9 + 48);
        if ( !(_DWORD)v11 )
          goto LABEL_4;
        v12 = *(_QWORD *)(v9 + 32);
        v13 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v14 = (__int64 *)(v12 + 16LL * v13);
        v15 = *v14;
        if ( v10 != *v14 )
        {
          v49 = 1;
          while ( v15 != -8 )
          {
            v50 = v49 + 1;
            v13 = (v11 - 1) & (v49 + v13);
            v14 = (__int64 *)(v12 + 16LL * v13);
            v15 = *v14;
            if ( v10 == *v14 )
              goto LABEL_7;
            v49 = v50;
          }
          goto LABEL_4;
        }
LABEL_7:
        if ( v14 == (__int64 *)(v12 + 16 * v11) )
          goto LABEL_4;
        if ( !v14[1] )
          goto LABEL_4;
        v16 = *(unsigned int *)(a1 + 80);
        if ( !(_DWORD)v16 )
          goto LABEL_4;
        v17 = (unsigned int)(v16 - 1);
        v18 = *(_QWORD *)(a1 + 64);
        v19 = v17 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v20 = (__int64 *)(v18 + 16LL * v19);
        v21 = *v20;
        if ( v10 != *v20 )
        {
          v51 = 1;
          while ( v21 != -8 )
          {
            v52 = v51 + 1;
            v19 = v17 & (v51 + v19);
            v20 = (__int64 *)(v18 + 16LL * v19);
            v21 = *v20;
            if ( v10 == *v20 )
              goto LABEL_11;
            v51 = v52;
          }
          goto LABEL_4;
        }
LABEL_11:
        if ( v20 == (__int64 *)(v18 + 16 * v16) )
          goto LABEL_4;
        v22 = *(_QWORD *)(v20[1] + 8);
        if ( !v22 )
          BUG();
        if ( *(_BYTE *)(v22 - 16) == 23 )
        {
          v23 = *(_QWORD *)(a1 + 120);
          v24 = v22 - 32;
          v25 = *(_DWORD *)(v22 - 12) & 0xFFFFFFF;
          if ( v25 == *(_DWORD *)(v22 + 44) )
          {
            v56 = *(_QWORD *)(a1 + 120);
            v57 = *(_QWORD *)(v20[1] + 8);
            v53 = v25 + (v25 >> 1);
            v58 = v22 - 32;
            if ( (unsigned int)v53 < 2 )
              v53 = 2;
            *(_DWORD *)(v22 + 44) = v53;
            sub_16488D0(v24, v53, 1, v22, v23, v17);
            v22 = v57;
            v23 = v56;
            v24 = v58;
            v25 = *(_DWORD *)(v57 - 12) & 0xFFFFFFF;
          }
          v26 = (v25 + 1) & 0xFFFFFFF;
          v27 = v26 | *(_DWORD *)(v22 - 12) & 0xF0000000;
          *(_DWORD *)(v22 - 12) = v27;
          if ( (v27 & 0x40000000) != 0 )
            v28 = *(_QWORD *)(v22 - 40);
          else
            v28 = v24 - 24 * v26;
          v29 = (_QWORD *)(v28 + 24LL * (unsigned int)(v26 - 1));
          if ( *v29 )
          {
            v30 = v29[1];
            v31 = v29[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v31 = v30;
            if ( v30 )
              *(_QWORD *)(v30 + 16) = *(_QWORD *)(v30 + 16) & 3LL | v31;
          }
          *v29 = v23;
          if ( v23 )
          {
            v32 = *(_QWORD *)(v23 + 8);
            v29[1] = v32;
            if ( v32 )
              *(_QWORD *)(v32 + 16) = (unsigned __int64)(v29 + 1) | *(_QWORD *)(v32 + 16) & 3LL;
            v29[2] = (v23 + 8) | v29[2] & 3LL;
            *(_QWORD *)(v23 + 8) = v29;
          }
          v33 = *(_DWORD *)(v22 - 12) & 0xFFFFFFF;
          if ( (*(_BYTE *)(v22 - 9) & 0x40) != 0 )
            v34 = *(_QWORD *)(v22 - 40);
          else
            v34 = v24 - 24 * v33;
          ++v7;
          *(_QWORD *)(v34 + 8LL * (unsigned int)(v33 - 1) + 24LL * *(unsigned int *)(v22 + 44) + 8) = a2;
          if ( v5 == v7 )
            break;
        }
        else
        {
LABEL_4:
          if ( v5 == ++v7 )
            break;
        }
      }
    }
  }
  result = *(unsigned int *)(a1 + 80);
  if ( (_DWORD)result )
  {
    v36 = *(_QWORD *)(a1 + 64);
    v37 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v38 = (__int64 *)(v36 + 16LL * v37);
    v39 = *v38;
    if ( a2 == *v38 )
    {
LABEL_29:
      result = v36 + 16 * result;
      if ( v38 != (__int64 *)result )
      {
        v40 = (unsigned __int64 *)v38[1];
        v41 = (unsigned __int64 *)v40[1];
        while ( v40 != v41 )
        {
          while ( 1 )
          {
            v47 = v41;
            v41 = (unsigned __int64 *)v41[1];
            if ( (unsigned int)*((unsigned __int8 *)v47 - 16) - 21 <= 1 )
              break;
            v48 = *v47 & 0xFFFFFFFFFFFFFFF8LL;
            *v41 = v48 | *v41 & 7;
            *(_QWORD *)(v48 + 8) = v41;
            *v47 &= 7u;
            v47[1] = 0;
            result = sub_164BEC0();
            if ( v40 == v41 )
              return result;
          }
          result = *(_QWORD *)(a1 + 120);
          if ( *(v47 - 7) )
          {
            v42 = *(v47 - 6);
            v43 = *(v47 - 5) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v43 = v42;
            if ( v42 )
              *(_QWORD *)(v42 + 16) = *(_QWORD *)(v42 + 16) & 3LL | v43;
          }
          *(v47 - 7) = result;
          if ( result )
          {
            v44 = *(_QWORD *)(result + 8);
            *(v47 - 6) = v44;
            if ( v44 )
              *(_QWORD *)(v44 + 16) = (unsigned __int64)(v47 - 6) | *(_QWORD *)(v44 + 16) & 3LL;
            v45 = *(v47 - 5);
            v46 = v47 - 7;
            v46[2] = (result + 8) | v45 & 3;
            *(_QWORD *)(result + 8) = v46;
          }
        }
      }
    }
    else
    {
      v54 = 1;
      while ( v39 != -8 )
      {
        v55 = v54 + 1;
        v37 = (result - 1) & (v54 + v37);
        v38 = (__int64 *)(v36 + 16LL * v37);
        v39 = *v38;
        if ( a2 == *v38 )
          goto LABEL_29;
        v54 = v55;
      }
    }
  }
  return result;
}
