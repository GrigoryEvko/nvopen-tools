// Function: sub_17E41E0
// Address: 0x17e41e0
//
__int64 __fastcall sub_17E41E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 v6; // r9
  unsigned int v7; // edi
  unsigned int v8; // r10d
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r11
  _QWORD *v12; // rdx
  _QWORD *v13; // rcx
  _QWORD ****v14; // r11
  _QWORD ***v15; // r9
  _QWORD **v16; // r10
  _QWORD *v17; // rbx
  _QWORD *****v18; // r12
  _QWORD ***v19; // rax
  unsigned int v20; // r8d
  __int64 *v21; // rax
  __int64 v22; // rdx
  _QWORD *v23; // rsi
  _QWORD *v24; // rdx
  _QWORD *********v25; // r8
  _QWORD ********v26; // r9
  _QWORD *******v27; // r10
  _QWORD ******v28; // r11
  _QWORD *****v29; // rbx
  _QWORD ****v30; // rdi
  _QWORD ***v31; // rax
  unsigned int v32; // eax
  int v34; // eax
  int v35; // r11d
  int v36; // eax
  int v37; // ebx

  v3 = a1;
  v6 = *(_QWORD *)(a1 + 40);
  v7 = *(_DWORD *)(a1 + 56);
  if ( v7 )
  {
    v8 = v7 - 1;
    v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
    {
LABEL_3:
      v12 = (_QWORD *)v10[1];
      v13 = (_QWORD *)*v12;
      if ( v12 == (_QWORD *)*v12 )
      {
LABEL_19:
        v20 = v8 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
        v21 = (__int64 *)(v6 + 16LL * v20);
        v22 = *v21;
        if ( a3 == *v21 )
          goto LABEL_20;
        v34 = 1;
        while ( v22 != -8 )
        {
          v35 = v34 + 1;
          v20 = v8 & (v34 + v20);
          v21 = (__int64 *)(v6 + 16LL * v20);
          v22 = *v21;
          if ( a3 == *v21 )
            goto LABEL_20;
          v34 = v35;
        }
        goto LABEL_41;
      }
    }
    else
    {
      v36 = 1;
      while ( v11 != -8 )
      {
        v37 = v36 + 1;
        v9 = v8 & (v36 + v9);
        v10 = (__int64 *)(v6 + 16LL * v9);
        v11 = *v10;
        if ( a2 == *v10 )
          goto LABEL_3;
        v36 = v37;
      }
      v12 = *(_QWORD **)(v6 + 16LL * v7 + 8);
      v13 = (_QWORD *)*v12;
      v14 = (_QWORD ****)v12;
      if ( v12 == (_QWORD *)*v12 )
        goto LABEL_18;
    }
  }
  else
  {
    v12 = *(_QWORD **)(v6 + 8);
    v13 = (_QWORD *)*v12;
    if ( (_QWORD *)*v12 == v12 )
    {
LABEL_41:
      v21 = (__int64 *)(v6 + 16LL * v7);
      goto LABEL_20;
    }
  }
  v14 = (_QWORD ****)*v13;
  if ( v13 != (_QWORD *)*v13 )
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
          v18 = (_QWORD *****)*v17;
          if ( v17 != (_QWORD *)*v17 )
          {
            if ( v18 != *v18 )
            {
              v19 = sub_17E2860(*v18);
              *v18 = (_QWORD ****)v19;
              v18 = (_QWORD *****)v19;
            }
            *v17 = v18;
          }
          *v16 = v18;
          v16 = v18;
        }
        *v15 = v16;
      }
      *v14 = (_QWORD ***)v16;
      v14 = (_QWORD ****)v16;
    }
    *v13 = v14;
  }
  *v12 = v14;
  v7 = *(_DWORD *)(v3 + 56);
  v6 = *(_QWORD *)(v3 + 40);
  if ( v7 )
  {
    v8 = v7 - 1;
LABEL_18:
    v13 = v14;
    goto LABEL_19;
  }
  v13 = v14;
  v21 = *(__int64 **)(v3 + 40);
LABEL_20:
  v23 = (_QWORD *)v21[1];
  v24 = (_QWORD *)*v23;
  if ( v23 != (_QWORD *)*v23 )
  {
    v25 = (_QWORD *********)*v24;
    if ( v24 != (_QWORD *)*v24 )
    {
      v26 = *v25;
      if ( v25 != *v25 )
      {
        v27 = *v26;
        if ( v26 != *v26 )
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
                v31 = sub_17E2860(v30);
                *v29 = (_QWORD ****)v31;
                v30 = (_QWORD ****)v31;
              }
              *v28 = (_QWORD *****)v30;
              v28 = (_QWORD ******)v30;
            }
            *v27 = v28;
          }
          *v26 = (_QWORD *******)v28;
          v26 = (_QWORD ********)v28;
        }
        *v25 = v26;
      }
      *v24 = v26;
      v24 = v26;
    }
    *v23 = v24;
  }
  if ( v24 == v13 )
    return 0;
  v32 = *((_DWORD *)v24 + 3);
  if ( *((_DWORD *)v13 + 3) < v32 )
  {
    *v13 = v24;
    return 1;
  }
  *v24 = v13;
  if ( v32 != *((_DWORD *)v13 + 3) )
    return 1;
  *((_DWORD *)v13 + 3) = v32 + 1;
  return 1;
}
