// Function: sub_21020C0
// Address: 0x21020c0
//
void __fastcall sub_21020C0(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4, __int64 *a5)
{
  unsigned int v7; // ebx
  __int64 v8; // r13
  _QWORD *v9; // rax
  __int64 v10; // r9
  __int64 v11; // rdi
  unsigned int v12; // r14d
  void (*v13)(); // rax
  _DWORD *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 *v17; // r13
  unsigned int v18; // r15d
  unsigned int v19; // r8d
  bool v20; // bl
  __int64 *v21; // r14
  __int64 v22; // rdx
  __int64 v23; // rdi
  void (*v24)(); // rax
  unsigned __int64 v25; // rsi
  _BYTE *v26; // rdx
  __int64 v27; // [rsp+0h] [rbp-180h]
  unsigned int v28; // [rsp+14h] [rbp-16Ch]
  __int64 v29; // [rsp+20h] [rbp-160h]
  int v31; // [rsp+30h] [rbp-150h]
  __int64 v32; // [rsp+38h] [rbp-148h]
  __int64 *v33; // [rsp+40h] [rbp-140h] BYREF
  __int64 v34; // [rsp+48h] [rbp-138h]
  _BYTE v35[64]; // [rsp+50h] [rbp-130h] BYREF
  __int64 v36; // [rsp+90h] [rbp-F0h] BYREF
  _BYTE *v37; // [rsp+98h] [rbp-E8h]
  _BYTE *v38; // [rsp+A0h] [rbp-E0h]
  __int64 v39; // [rsp+A8h] [rbp-D8h]
  int v40; // [rsp+B0h] [rbp-D0h]
  _BYTE v41[64]; // [rsp+B8h] [rbp-C8h] BYREF
  _BYTE *v42; // [rsp+F8h] [rbp-88h]
  __int64 v43; // [rsp+100h] [rbp-80h]
  _BYTE v44[120]; // [rsp+108h] [rbp-78h] BYREF

  v37 = v41;
  v38 = v41;
  v42 = v44;
  v43 = 0x800000000LL;
  v31 = a4;
  v32 = (__int64)a5;
  v36 = 0;
  v39 = 8;
  v40 = 0;
  v29 = (unsigned int)(a4 - 1);
LABEL_2:
  while ( 1 )
  {
    v7 = *(_DWORD *)(a2 + 8);
    if ( !v7 )
      break;
LABEL_30:
    v25 = *(_QWORD *)(*(_QWORD *)a2 + 8LL * v7 - 8);
    *(_DWORD *)(a2 + 8) = v7 - 1;
    sub_2100DF0((_QWORD *)a1, v25, (__int64)&v36, v32);
  }
  while ( (_DWORD)v43 )
  {
    v8 = *(_QWORD *)&v42[8 * (unsigned int)v43 - 8];
    v9 = v37;
    if ( v38 == v37 )
    {
      v26 = &v37[8 * HIDWORD(v39)];
      if ( v37 == v26 )
      {
LABEL_39:
        v9 = &v37[8 * HIDWORD(v39)];
      }
      else
      {
        while ( v8 != *v9 )
        {
          if ( v26 == (_BYTE *)++v9 )
            goto LABEL_39;
        }
      }
    }
    else
    {
      v9 = sub_16CC9F0((__int64)&v36, v8);
      if ( v8 == *v9 )
      {
        if ( v38 == v37 )
        {
          a4 = HIDWORD(v39);
          v26 = &v38[8 * HIDWORD(v39)];
        }
        else
        {
          a4 = (unsigned int)v39;
          v26 = &v38[8 * (unsigned int)v39];
        }
      }
      else
      {
        if ( v38 != v37 )
          goto LABEL_7;
        v9 = &v38[8 * HIDWORD(v39)];
        v26 = v9;
      }
    }
    if ( v9 != (_QWORD *)v26 )
    {
      *v9 = -2;
      ++v40;
    }
LABEL_7:
    LODWORD(v43) = v43 - 1;
    if ( (unsigned __int8)sub_2101C50(a1, v8, a2, a4, (__int64)a5) )
      goto LABEL_2;
    v11 = *(_QWORD *)(a1 + 56);
    v12 = *(_DWORD *)(v8 + 112);
    if ( v11 )
    {
      v13 = *(void (**)())(*(_QWORD *)v11 + 40LL);
      if ( v13 != nullsub_747 )
        ((void (__fastcall *)(__int64, _QWORD))v13)(v11, v12);
    }
    if ( !(unsigned __int8)sub_1DC0580(*(_QWORD **)(a1 + 32), v8, a2, a4, (__int64)a5, v10) )
      goto LABEL_2;
    if ( v31 )
    {
      a4 = (__int64)a3;
      v14 = a3;
      while ( *v14 != v12 )
      {
        if ( ++v14 == &a3[v29 + 1] )
          goto LABEL_16;
      }
      goto LABEL_2;
    }
LABEL_16:
    sub_1DB4280((__int64 *)v8);
    v15 = *(_QWORD *)(a1 + 32);
    v33 = (__int64 *)v35;
    v34 = 0x800000000LL;
    sub_1DBEB50(v15, v8, (__int64)&v33);
    v16 = *(_QWORD *)(a1 + 40);
    if ( v16 )
    {
      v7 = *(_DWORD *)(*(_QWORD *)(v16 + 312) + 4LL * (v12 & 0x7FFFFFFF));
      if ( !v7 )
        v7 = v12;
    }
    v17 = v33;
    a5 = &v33[(unsigned int)v34];
    if ( v33 != a5 )
    {
      v27 = a2;
      v18 = v7;
      v19 = v12;
      v20 = v7 != 0 && v7 != v12;
      v21 = &v33[(unsigned int)v34];
      do
      {
        while ( 1 )
        {
          v22 = *v17;
          if ( v20 )
            *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 312LL) + 4LL * (*(_DWORD *)(v22 + 112) & 0x7FFFFFFF)) = v18;
          v23 = *(_QWORD *)(a1 + 56);
          if ( v23 )
          {
            v24 = *(void (**)())(*(_QWORD *)v23 + 48LL);
            if ( v24 != nullsub_739 )
              break;
          }
          if ( v21 == ++v17 )
            goto LABEL_27;
        }
        ++v17;
        v28 = v19;
        ((void (__fastcall *)(__int64, _QWORD, _QWORD))v24)(v23, *(unsigned int *)(v22 + 112), v19);
        v19 = v28;
      }
      while ( v21 != v17 );
LABEL_27:
      a2 = v27;
      a5 = v33;
    }
    if ( a5 == (__int64 *)v35 )
      goto LABEL_2;
    _libc_free((unsigned __int64)a5);
    v7 = *(_DWORD *)(a2 + 8);
    if ( v7 )
      goto LABEL_30;
  }
  if ( v42 != v44 )
    _libc_free((unsigned __int64)v42);
  if ( v38 != v37 )
    _libc_free((unsigned __int64)v38);
}
