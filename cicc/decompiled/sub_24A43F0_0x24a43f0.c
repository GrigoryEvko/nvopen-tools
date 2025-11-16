// Function: sub_24A43F0
// Address: 0x24a43f0
//
__int64 __fastcall sub_24A43F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  unsigned int v6; // r9d
  __int64 v7; // rdi
  unsigned int v8; // r10d
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r11
  _QWORD **********v12; // rcx
  _QWORD ***********v13; // r11
  _QWORD **********v14; // rdx
  _QWORD *********v15; // r9
  _QWORD ********v16; // r10
  _QWORD *******v17; // rbx
  _QWORD ******v18; // r12
  _QWORD *****v19; // r13
  _QWORD ****v20; // rdi
  _QWORD ***v21; // rax
  unsigned int v22; // r8d
  __int64 *v23; // rax
  __int64 v24; // rdx
  _QWORD *v25; // rsi
  _QWORD *v26; // rdx
  _QWORD *****v27; // r8
  _QWORD ****v28; // r9
  _QWORD ***v29; // r10
  _QWORD **v30; // r11
  _QWORD *v31; // rbx
  _QWORD *****v32; // r12
  _QWORD ***v33; // rax
  unsigned int v34; // eax
  int v36; // edx
  int v37; // eax
  int v38; // r11d
  int v39; // ebx

  v3 = a1;
  v6 = *(_DWORD *)(a1 + 56);
  v7 = *(_QWORD *)(a1 + 40);
  if ( !v6 )
  {
    v12 = *(_QWORD ***********)(v7 + 8);
    v13 = (_QWORD ***********)*v12;
    if ( *v12 == v12 )
      goto LABEL_45;
    goto LABEL_4;
  }
  v8 = v6 - 1;
  v9 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( a2 != *v10 )
  {
    v36 = 1;
    while ( v11 != -4096 )
    {
      v39 = v36 + 1;
      v9 = v8 & (v36 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        goto LABEL_3;
      v36 = v39;
    }
    v12 = *(_QWORD ***********)(v7 + 16LL * v6 + 8);
    v13 = (_QWORD ***********)*v12;
    v14 = (_QWORD **********)*v12;
    if ( *v12 == v12 )
    {
LABEL_20:
      v12 = v14;
      goto LABEL_21;
    }
LABEL_4:
    v14 = *v13;
    if ( v13 != *v13 )
    {
      v15 = *v14;
      if ( v14 != *v14 )
      {
        v16 = *v15;
        if ( v15 != *v15 )
        {
          v17 = *v16;
          if ( v16 != *v16 )
          {
            v18 = *v17;
            if ( v17 != *v17 )
            {
              v19 = *v18;
              if ( v18 != *v18 )
              {
                v20 = *v19;
                if ( v19 != *v19 )
                {
                  v21 = sub_24A34D0(v20);
                  *v19 = (_QWORD ****)v21;
                  v20 = (_QWORD ****)v21;
                }
                *v18 = (_QWORD *****)v20;
                v18 = (_QWORD ******)v20;
              }
              *v17 = v18;
            }
            *v16 = (_QWORD *******)v18;
            v16 = (_QWORD ********)v18;
          }
          *v15 = v16;
        }
        *v14 = (_QWORD *********)v16;
        v14 = (_QWORD **********)v16;
      }
      *v13 = v14;
    }
    *v12 = v14;
    v6 = *(_DWORD *)(v3 + 56);
    v7 = *(_QWORD *)(v3 + 40);
    if ( !v6 )
    {
      v12 = v14;
      goto LABEL_45;
    }
    v8 = v6 - 1;
    goto LABEL_20;
  }
LABEL_3:
  v12 = (_QWORD **********)v10[1];
  v13 = (_QWORD ***********)*v12;
  if ( *v12 != v12 )
    goto LABEL_4;
LABEL_21:
  v22 = v8 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v23 = (__int64 *)(v7 + 16LL * v22);
  v24 = *v23;
  if ( a3 == *v23 )
    goto LABEL_22;
  v37 = 1;
  while ( v24 != -4096 )
  {
    v38 = v37 + 1;
    v22 = v8 & (v37 + v22);
    v23 = (__int64 *)(v7 + 16LL * v22);
    v24 = *v23;
    if ( a3 == *v23 )
      goto LABEL_22;
    v37 = v38;
  }
LABEL_45:
  v23 = (__int64 *)(v7 + 16LL * v6);
LABEL_22:
  v25 = (_QWORD *)v23[1];
  v26 = (_QWORD *)*v25;
  if ( v25 != (_QWORD *)*v25 )
  {
    v27 = (_QWORD *****)*v26;
    if ( v26 != (_QWORD *)*v26 )
    {
      v28 = *v27;
      if ( v27 != *v27 )
      {
        v29 = *v28;
        if ( v28 != *v28 )
        {
          v30 = *v29;
          if ( v29 != *v29 )
          {
            v31 = *v30;
            if ( v30 != *v30 )
            {
              v32 = (_QWORD *****)*v31;
              if ( v31 != (_QWORD *)*v31 )
              {
                if ( v32 != *v32 )
                {
                  v33 = sub_24A34D0(*v32);
                  *v32 = (_QWORD ****)v33;
                  v32 = (_QWORD *****)v33;
                }
                *v31 = v32;
              }
              *v30 = v32;
              v30 = v32;
            }
            *v29 = v30;
          }
          *v28 = (_QWORD ***)v30;
          v28 = (_QWORD ****)v30;
        }
        *v27 = v28;
      }
      *v26 = v28;
      v26 = v28;
    }
    *v25 = v26;
  }
  if ( v12 == v26 )
    return 0;
  v34 = *((_DWORD *)v26 + 3);
  if ( *((_DWORD *)v12 + 3) < v34 )
  {
    *v12 = (_QWORD *********)v26;
    return 1;
  }
  *v26 = v12;
  if ( v34 != *((_DWORD *)v12 + 3) )
    return 1;
  *((_DWORD *)v12 + 3) = v34 + 1;
  return 1;
}
