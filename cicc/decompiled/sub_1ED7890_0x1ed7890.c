// Function: sub_1ED7890
// Address: 0x1ed7890
//
void __fastcall sub_1ED7890(__int64 a1, __int64 **a2)
{
  __int64 v3; // r9
  __int64 *v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r10
  __int64 v8; // r11
  unsigned int v9; // r8d
  int v10; // eax
  __int16 v11; // r15
  __int64 v12; // rbx
  int v13; // r14d
  __int64 v14; // rdx
  __int64 v15; // rsi
  bool v16; // zf
  unsigned __int16 v17; // r12
  unsigned int v18; // ecx
  __int64 v19; // rdx
  char v20; // r12
  int v21; // edi
  __int64 v22; // rax
  __int64 v23; // r8
  int v24; // edx
  unsigned int v25; // edx
  __int64 v26; // rdi
  __int64 (__fastcall *v27)(__int64, __int64); // rax
  __int64 v28; // rax
  _DWORD *v29; // rbx
  __int64 v30; // rax
  __int64 v31; // rdi
  __int64 v32; // [rsp+0h] [rbp-90h]
  __int64 v33; // [rsp+8h] [rbp-88h]
  int v34; // [rsp+10h] [rbp-80h]
  unsigned int v35; // [rsp+14h] [rbp-7Ch]
  __int64 v36; // [rsp+18h] [rbp-78h]
  __int64 v37; // [rsp+18h] [rbp-78h]
  __int64 v38; // [rsp+18h] [rbp-78h]
  __int64 v39; // [rsp+20h] [rbp-70h]
  _BYTE *v41; // [rsp+30h] [rbp-60h] BYREF
  __int64 v42; // [rsp+38h] [rbp-58h]
  _BYTE v43[80]; // [rsp+40h] [rbp-50h] BYREF

  v3 = *(_QWORD *)a1 + 24LL * *((unsigned __int16 *)*a2 + 12);
  if ( !*(_QWORD *)(v3 + 16) )
  {
    v39 = *(_QWORD *)a1 + 24LL * *((unsigned __int16 *)*a2 + 12);
    v30 = sub_2207820(2LL * *((unsigned __int16 *)*a2 + 10));
    v3 = v39;
    v31 = *(_QWORD *)(v39 + 16);
    *(_QWORD *)(v39 + 16) = v30;
    if ( v31 )
    {
      j_j___libc_free_0_0(v31);
      v3 = v39;
    }
  }
  v41 = v43;
  v42 = 0x1000000000LL;
  v4 = a2[5];
  if ( v4 )
  {
    v36 = v3;
    v5 = ((__int64 (__fastcall *)(_QWORD))v4)(*(_QWORD *)(a1 + 16));
    v3 = v36;
    v7 = v6;
    v8 = v5;
    if ( v6 )
    {
LABEL_4:
      v9 = 255;
      v10 = 0;
      v11 = 0;
      v12 = 0;
      v13 = -1;
      v14 = 0;
      while ( 1 )
      {
        while ( 1 )
        {
          v17 = *(_WORD *)(v8 + 2 * v14);
          if ( (*(_QWORD *)(*(_QWORD *)(a1 + 64) + 8 * ((unsigned __int64)v17 >> 6)) & (1LL << v17)) == 0 )
            break;
LABEL_8:
          v14 = (unsigned int)++v10;
          if ( v10 == v7 )
            goto LABEL_16;
        }
        v18 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 232LL) + 8LL * v17);
        if ( v9 > v18 )
          v9 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 232LL) + 8LL * v17);
        if ( !*(_WORD *)(*(_QWORD *)(a1 + 40) + 2LL * v17) )
        {
          v15 = (unsigned int)v12;
          v16 = v18 == v13;
          v13 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 232LL) + 8LL * v17);
          if ( !v16 )
            v11 = v12;
          v12 = (unsigned int)(v12 + 1);
          *(_WORD *)(*(_QWORD *)(v3 + 16) + 2 * v15) = v17;
          goto LABEL_8;
        }
        v19 = (unsigned int)v42;
        if ( (unsigned int)v42 >= HIDWORD(v42) )
        {
          v34 = v10;
          v32 = v3;
          v33 = v7;
          v35 = v9;
          v38 = v8;
          sub_16CD150((__int64)&v41, v43, 0, 2, v9, v3);
          v19 = (unsigned int)v42;
          v10 = v34;
          v3 = v32;
          v7 = v33;
          v9 = v35;
          v8 = v38;
        }
        *(_WORD *)&v41[2 * v19] = v17;
        LODWORD(v42) = v42 + 1;
        v14 = (unsigned int)++v10;
        if ( v10 == v7 )
        {
LABEL_16:
          v20 = v9;
          goto LABEL_17;
        }
      }
    }
  }
  else
  {
    v7 = *((unsigned __int16 *)*a2 + 10);
    v8 = **a2;
    if ( *((_WORD *)*a2 + 10) )
      goto LABEL_4;
  }
  v20 = -1;
  v11 = 0;
  v13 = -1;
  v12 = 0;
LABEL_17:
  *(_DWORD *)(v3 + 4) = v12 + v42;
  if ( (_DWORD)v42 )
  {
    v21 = v12 + v42;
    v22 = 0;
    while ( 1 )
    {
      v23 = *(unsigned __int16 *)&v41[v22];
      v24 = v13;
      v13 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 232LL) + 8 * v23);
      v16 = v13 == v24;
      v25 = v12 + 1;
      *(_WORD *)(*(_QWORD *)(v3 + 16) + 2 * v12) = v23;
      if ( !v16 )
        v11 = v12;
      v22 += 2;
      if ( v25 == v21 )
        break;
      v12 = v25;
    }
  }
  if ( dword_4FC9D60 && *(_DWORD *)(v3 + 4) > (unsigned int)dword_4FC9D60 )
    *(_DWORD *)(v3 + 4) = dword_4FC9D60;
  v26 = *(_QWORD *)(a1 + 24);
  v27 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v26 + 160LL);
  if ( v27 != sub_1E693B0 )
  {
    v37 = v3;
    v28 = ((__int64 (__fastcall *)(__int64, __int64 **, _QWORD))v27)(v26, a2, *(_QWORD *)(a1 + 16));
    v3 = v37;
    if ( v28 )
    {
      if ( a2 != (__int64 **)v28 )
      {
        v29 = (_DWORD *)(*(_QWORD *)a1 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v28 + 24LL));
        if ( *(_DWORD *)(a1 + 8) == *v29 )
        {
          if ( *(_DWORD *)(v37 + 4) >= v29[1] )
            goto LABEL_27;
          goto LABEL_36;
        }
        sub_1ED7890(a1);
        v3 = v37;
        if ( *(_DWORD *)(v37 + 4) < v29[1] )
LABEL_36:
          *(_BYTE *)(v3 + 8) = 1;
      }
    }
  }
LABEL_27:
  *(_BYTE *)(v3 + 9) = v20;
  *(_WORD *)(v3 + 10) = v11;
  *(_DWORD *)v3 = *(_DWORD *)(a1 + 8);
  if ( v41 != v43 )
    _libc_free((unsigned __int64)v41);
}
