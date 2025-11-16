// Function: sub_1F15810
// Address: 0x1f15810
//
void __fastcall sub_1F15810(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v10; // r15
  int v11; // r11d
  unsigned __int64 v12; // rdx
  unsigned int v13; // eax
  __int64 v14; // r10
  __int64 v15; // rbx
  __int64 v16; // r14
  __int64 v17; // rbx
  __int64 v18; // r15
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  __int64 v21; // r13
  __int64 v22; // rax
  unsigned int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // r14
  __int64 v26; // rdi
  _QWORD *v27; // rsi
  _QWORD *v28; // rdx
  __int64 v29; // [rsp+0h] [rbp-B0h]
  unsigned int v30; // [rsp+Ch] [rbp-A4h]
  __int64 v31; // [rsp+18h] [rbp-98h]
  __int64 v32; // [rsp+20h] [rbp-90h]
  int v33; // [rsp+20h] [rbp-90h]
  int *v34; // [rsp+28h] [rbp-88h]
  _BYTE *v35; // [rsp+30h] [rbp-80h] BYREF
  __int64 v36; // [rsp+38h] [rbp-78h]
  _BYTE v37[112]; // [rsp+40h] [rbp-70h] BYREF

  v6 = *(_QWORD *)(a1 + 72);
  v35 = v37;
  v7 = *(_QWORD *)(v6 + 16);
  v8 = *(unsigned int *)(v6 + 64);
  v36 = 0x800000000LL;
  v34 = (int *)(*(_QWORD *)v7 + 4 * v8);
  v31 = *(_QWORD *)v7 + 4LL * *(unsigned int *)(v7 + 8);
  if ( (int *)v31 == v34 )
    return;
  do
  {
    v10 = *(_QWORD *)(a1 + 16);
    v11 = *v34;
    v12 = *(unsigned int *)(v10 + 408);
    v13 = *v34 & 0x7FFFFFFF;
    v14 = v13;
    v15 = 8LL * v13;
    if ( v13 >= (unsigned int)v12 || (v16 = *(_QWORD *)(*(_QWORD *)(v10 + 400) + 8LL * v13)) == 0 )
    {
      v23 = v13 + 1;
      if ( (unsigned int)v12 < v23 )
      {
        v25 = v23;
        if ( v23 >= v12 )
        {
          if ( v23 > v12 )
          {
            if ( v23 > (unsigned __int64)*(unsigned int *)(v10 + 412) )
            {
              v29 = v14;
              v30 = v23;
              v33 = *v34;
              sub_16CD150(v10 + 400, (const void *)(v10 + 416), v23, 8, a5, a6);
              v12 = *(unsigned int *)(v10 + 408);
              v14 = v29;
              v23 = v30;
              v11 = v33;
            }
            v24 = *(_QWORD *)(v10 + 400);
            v26 = *(_QWORD *)(v10 + 416);
            v27 = (_QWORD *)(v24 + 8 * v25);
            v28 = (_QWORD *)(v24 + 8 * v12);
            if ( v27 != v28 )
            {
              do
                *v28++ = v26;
              while ( v27 != v28 );
              v24 = *(_QWORD *)(v10 + 400);
            }
            *(_DWORD *)(v10 + 408) = v23;
            goto LABEL_22;
          }
        }
        else
        {
          *(_DWORD *)(v10 + 408) = v23;
        }
      }
      v24 = *(_QWORD *)(v10 + 400);
LABEL_22:
      v32 = v14;
      *(_QWORD *)(v24 + v15) = sub_1DBA290(v11);
      v16 = *(_QWORD *)(*(_QWORD *)(v10 + 400) + 8 * v32);
      sub_1DBB110((_QWORD *)v10, v16);
    }
    v17 = *(_QWORD *)v16;
    v18 = *(_QWORD *)v16 + 24LL * *(unsigned int *)(v16 + 8);
    if ( *(_QWORD *)v16 != v18 )
    {
      do
      {
        v19 = *(_QWORD *)(*(_QWORD *)(v17 + 16) + 8LL);
        v20 = v19 & 0xFFFFFFFFFFFFFFF8LL;
        if ( *(_QWORD *)(v17 + 8) == (v19 & 0xFFFFFFFFFFFFFFF8LL | 6) && (v19 & 6) != 0 )
        {
          v21 = 0;
          if ( v20 )
            v21 = *(_QWORD *)(v20 + 16);
          sub_1E1B440(v21, *(_DWORD *)(v16 + 112), *(_QWORD **)(a1 + 56), 0, a5, a6);
          if ( (unsigned __int8)sub_1E17E50(v21) )
          {
            v22 = (unsigned int)v36;
            if ( (unsigned int)v36 >= HIDWORD(v36) )
            {
              sub_16CD150((__int64)&v35, v37, 0, 8, a5, a6);
              v22 = (unsigned int)v36;
            }
            *(_QWORD *)&v35[8 * v22] = v21;
            LODWORD(v36) = v36 + 1;
          }
        }
        v17 += 24;
      }
      while ( v18 != v17 );
    }
    ++v34;
  }
  while ( v34 != (int *)v31 );
  if ( (_DWORD)v36 )
    sub_21020C0(*(_QWORD *)(a1 + 72), &v35, 0, 0, *(_QWORD *)(a1 + 8));
  if ( v35 != v37 )
    _libc_free((unsigned __int64)v35);
}
