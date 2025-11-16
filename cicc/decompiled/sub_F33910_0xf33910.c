// Function: sub_F33910
// Address: 0xf33910
//
__int64 __fastcall sub_F33910(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, char a6)
{
  __int64 *v6; // r12
  __int64 *v7; // rbx
  __int64 v8; // r9
  __int64 *v9; // rax
  __int64 result; // rax
  __int64 v11; // rdi
  __int64 v12; // r12
  __int64 v13; // r8
  unsigned int v14; // ecx
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // r15
  __int64 v19; // r14
  __int64 v20; // r13
  __int64 v21; // rcx
  __int64 v22; // rsi
  __int64 *v23; // rax
  __int64 *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r14
  int v27; // eax
  int v28; // eax
  unsigned int v29; // ecx
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rcx
  __int64 *v33; // rax
  __int64 *v34; // rdx
  __int64 v35; // r14
  __int64 v36; // rax
  __int64 v37; // r15
  __int64 v38; // rdi
  __int64 v39; // r14
  __int64 v40; // r13
  __int64 *v41; // rax
  __int64 *v42; // rdx
  __int64 v43; // rdx
  int v44; // eax
  int v45; // eax
  unsigned int v46; // ecx
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rcx
  int v51; // eax
  int v52; // eax
  unsigned int v53; // edx
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 *v57; // rax
  int v58; // [rsp+8h] [rbp-148h]
  __int64 *v60; // [rsp+18h] [rbp-138h]
  __int64 v61; // [rsp+20h] [rbp-130h]
  __int64 v62; // [rsp+28h] [rbp-128h]
  __int64 v64; // [rsp+48h] [rbp-108h] BYREF
  const char *v65; // [rsp+50h] [rbp-100h] BYREF
  __int64 *v66; // [rsp+58h] [rbp-F8h]
  char *v67; // [rsp+60h] [rbp-F0h]
  __int16 v68; // [rsp+70h] [rbp-E0h]
  __int64 v69; // [rsp+80h] [rbp-D0h] BYREF
  __int64 *v70; // [rsp+88h] [rbp-C8h]
  __int64 v71; // [rsp+90h] [rbp-C0h]
  int v72; // [rsp+98h] [rbp-B8h]
  unsigned __int8 v73; // [rsp+9Ch] [rbp-B4h]
  char v74; // [rsp+A0h] [rbp-B0h] BYREF

  v6 = &a3[a4];
  v7 = a3;
  v8 = 1;
  v61 = a2;
  v60 = a3;
  v58 = a4;
  v69 = 0;
  v70 = (__int64 *)&v74;
  v71 = 16;
  v72 = 0;
  v73 = 1;
  if ( v6 != a3 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        a2 = *v7;
        if ( (_BYTE)v8 )
          break;
LABEL_72:
        ++v7;
        sub_C8CC70((__int64)&v69, a2, (__int64)a3, a4, a5, v8);
        v8 = v73;
        if ( v6 == v7 )
          goto LABEL_8;
      }
      v9 = v70;
      a4 = HIDWORD(v71);
      a3 = &v70[HIDWORD(v71)];
      if ( v70 == a3 )
      {
LABEL_74:
        if ( HIDWORD(v71) >= (unsigned int)v71 )
          goto LABEL_72;
        a4 = (unsigned int)(HIDWORD(v71) + 1);
        ++v7;
        ++HIDWORD(v71);
        *a3 = a2;
        v8 = v73;
        ++v69;
        if ( v6 == v7 )
          break;
      }
      else
      {
        while ( a2 != *v9 )
        {
          if ( a3 == ++v9 )
            goto LABEL_74;
        }
        if ( v6 == ++v7 )
          break;
      }
    }
  }
LABEL_8:
  result = *(_QWORD *)(a1 + 56);
LABEL_9:
  if ( !result )
    BUG();
  v11 = result - 24;
  if ( *(_BYTE *)(result - 24) == 84 )
  {
    v12 = *(_QWORD *)(result + 8);
    v64 = result - 24;
    if ( !a6 )
    {
      v13 = *(_QWORD *)(result - 32);
      v14 = *(_DWORD *)(result - 20) & 0x7FFFFFF;
      if ( v14 )
      {
        v15 = *(unsigned int *)(result + 48);
        v16 = 0;
        v17 = v13 + 32 * v15;
        do
        {
          if ( *v60 == *(_QWORD *)(v17 + 8 * v16) )
          {
            v18 = *(_QWORD *)(v13 + 32 * v16);
            goto LABEL_17;
          }
          ++v16;
        }
        while ( v14 != (_DWORD)v16 );
        v18 = *(_QWORD *)(v13 + 0x1FFFFFFFE0LL);
LABEL_17:
        v19 = 0;
        v20 = 8LL * v14;
        while ( 2 )
        {
          v21 = *(_QWORD *)(v11 - 8);
          v22 = *(_QWORD *)(v21 + 32LL * *(unsigned int *)(v11 + 72) + v19);
          if ( v73 )
          {
            v23 = v70;
            v24 = &v70[HIDWORD(v71)];
            if ( v70 == v24 )
            {
LABEL_26:
              v19 += 8;
              if ( v19 == v20 )
                goto LABEL_27;
              continue;
            }
            while ( v22 != *v23 )
            {
              if ( v24 == ++v23 )
                goto LABEL_26;
            }
            v25 = *(_QWORD *)(v21 + 4 * v19);
            if ( v18 )
            {
LABEL_24:
              if ( !v25 || v18 != v25 )
                goto LABEL_40;
              goto LABEL_26;
            }
          }
          else
          {
            v33 = sub_C8CA60((__int64)&v69, v22);
            v11 = v64;
            if ( !v33 )
              goto LABEL_26;
            v25 = *(_QWORD *)(*(_QWORD *)(v64 - 8) + 4 * v19);
            if ( v18 )
              goto LABEL_24;
          }
          break;
        }
        v18 = v25;
        goto LABEL_26;
      }
      v18 = *(_QWORD *)(v13 + 0x1FFFFFFFE0LL);
LABEL_27:
      if ( v18 )
      {
        v65 = (const char *)&v69;
        v66 = &v64;
        sub_B57920(v11, (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_F33890, (__int64)&v65, 0);
        v26 = v64;
        v27 = *(_DWORD *)(v64 + 4) & 0x7FFFFFF;
        if ( v27 == *(_DWORD *)(v64 + 72) )
        {
          sub_B48D90(v64);
          v27 = *(_DWORD *)(v26 + 4) & 0x7FFFFFF;
        }
        v28 = (v27 + 1) & 0x7FFFFFF;
        v29 = v28 | *(_DWORD *)(v26 + 4) & 0xF8000000;
        v30 = *(_QWORD *)(v26 - 8) + 32LL * (unsigned int)(v28 - 1);
        *(_DWORD *)(v26 + 4) = v29;
        if ( *(_QWORD *)v30 )
        {
          v31 = *(_QWORD *)(v30 + 8);
          **(_QWORD **)(v30 + 16) = v31;
          if ( v31 )
            *(_QWORD *)(v31 + 16) = *(_QWORD *)(v30 + 16);
        }
        *(_QWORD *)v30 = v18;
        v32 = *(_QWORD *)(v18 + 16);
        a2 = v18 + 16;
        *(_QWORD *)(v30 + 8) = v32;
        if ( v32 )
          *(_QWORD *)(v32 + 16) = v30 + 8;
        *(_QWORD *)(v30 + 16) = a2;
        *(_QWORD *)(v18 + 16) = v30;
        *(_QWORD *)(*(_QWORD *)(v26 - 8)
                  + 32LL * *(unsigned int *)(v26 + 72)
                  + 8LL * ((*(_DWORD *)(v26 + 4) & 0x7FFFFFFu) - 1)) = v61;
        goto LABEL_36;
      }
    }
LABEL_40:
    v65 = sub_BD5D20(v11);
    v67 = ".ph";
    v68 = 773;
    v66 = v34;
    v35 = *(_QWORD *)(v64 + 8);
    v36 = sub_BD2DA0(80);
    v37 = v36;
    if ( v36 )
    {
      sub_B44260(v36, v35, 55, 0x8000000u, a5 + 24, 0);
      *(_DWORD *)(v37 + 72) = v58;
      sub_BD6B50((unsigned __int8 *)v37, &v65);
      sub_BD2A10(v37, *(_DWORD *)(v37 + 72), 1);
    }
    v38 = v64;
    a2 = *(unsigned int *)(v64 + 72);
    v39 = (*(_DWORD *)(v64 + 4) & 0x7FFFFFFu) - 1;
    while ( 1 )
    {
      v40 = *(_QWORD *)(*(_QWORD *)(v38 - 8) + 32LL * (unsigned int)a2 + 8 * v39);
      if ( v73 )
        break;
      v57 = sub_C8CA60((__int64)&v69, v40);
      v38 = v64;
      if ( v57 )
        goto LABEL_48;
LABEL_58:
      a2 = *(unsigned int *)(v38 + 72);
LABEL_59:
      if ( v39-- == 0 )
      {
        v51 = *(_DWORD *)(v38 + 4) & 0x7FFFFFF;
        if ( v51 == (_DWORD)a2 )
        {
          sub_B48D90(v38);
          v51 = *(_DWORD *)(v38 + 4) & 0x7FFFFFF;
        }
        v52 = (v51 + 1) & 0x7FFFFFF;
        v53 = v52 | *(_DWORD *)(v38 + 4) & 0xF8000000;
        v54 = *(_QWORD *)(v38 - 8) + 32LL * (unsigned int)(v52 - 1);
        *(_DWORD *)(v38 + 4) = v53;
        if ( *(_QWORD *)v54 )
        {
          v55 = *(_QWORD *)(v54 + 8);
          **(_QWORD **)(v54 + 16) = v55;
          if ( v55 )
            *(_QWORD *)(v55 + 16) = *(_QWORD *)(v54 + 16);
        }
        *(_QWORD *)v54 = v37;
        if ( v37 )
        {
          v56 = *(_QWORD *)(v37 + 16);
          *(_QWORD *)(v54 + 8) = v56;
          if ( v56 )
          {
            a2 = v54 + 8;
            *(_QWORD *)(v56 + 16) = v54 + 8;
          }
          *(_QWORD *)(v54 + 16) = v37 + 16;
          *(_QWORD *)(v37 + 16) = v54;
        }
        *(_QWORD *)(*(_QWORD *)(v38 - 8)
                  + 32LL * *(unsigned int *)(v38 + 72)
                  + 8LL * ((*(_DWORD *)(v38 + 4) & 0x7FFFFFFu) - 1)) = v61;
LABEL_36:
        result = v12;
        goto LABEL_9;
      }
    }
    v41 = v70;
    v42 = &v70[HIDWORD(v71)];
    if ( v70 == v42 )
      goto LABEL_59;
    while ( v40 != *v41 )
    {
      if ( v42 == ++v41 )
        goto LABEL_59;
    }
LABEL_48:
    v43 = sub_B48BF0(v38, v39, 0);
    v44 = *(_DWORD *)(v37 + 4) & 0x7FFFFFF;
    if ( v44 == *(_DWORD *)(v37 + 72) )
    {
      v62 = v43;
      sub_B48D90(v37);
      v43 = v62;
      v44 = *(_DWORD *)(v37 + 4) & 0x7FFFFFF;
    }
    v45 = (v44 + 1) & 0x7FFFFFF;
    v46 = v45 | *(_DWORD *)(v37 + 4) & 0xF8000000;
    v47 = *(_QWORD *)(v37 - 8) + 32LL * (unsigned int)(v45 - 1);
    *(_DWORD *)(v37 + 4) = v46;
    if ( *(_QWORD *)v47 )
    {
      v48 = *(_QWORD *)(v47 + 8);
      **(_QWORD **)(v47 + 16) = v48;
      if ( v48 )
        *(_QWORD *)(v48 + 16) = *(_QWORD *)(v47 + 16);
    }
    *(_QWORD *)v47 = v43;
    if ( v43 )
    {
      v49 = *(_QWORD *)(v43 + 16);
      *(_QWORD *)(v47 + 8) = v49;
      if ( v49 )
        *(_QWORD *)(v49 + 16) = v47 + 8;
      *(_QWORD *)(v47 + 16) = v43 + 16;
      *(_QWORD *)(v43 + 16) = v47;
    }
    *(_QWORD *)(*(_QWORD *)(v37 - 8)
              + 32LL * *(unsigned int *)(v37 + 72)
              + 8LL * ((*(_DWORD *)(v37 + 4) & 0x7FFFFFFu) - 1)) = v40;
    v38 = v64;
    goto LABEL_58;
  }
  if ( !v73 )
    return _libc_free(v70, a2);
  return result;
}
