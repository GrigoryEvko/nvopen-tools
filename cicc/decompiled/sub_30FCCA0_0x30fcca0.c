// Function: sub_30FCCA0
// Address: 0x30fcca0
//
void __fastcall sub_30FCCA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v7; // zf
  _QWORD *v8; // rax
  __int64 v9; // rdx
  _QWORD *v10; // r13
  __int64 v11; // rdx
  _QWORD *v12; // rbx
  __int64 *v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // r15
  __int64 *v16; // r12
  __int64 v17; // rbx
  _QWORD *v18; // rax
  _QWORD *v19; // rax
  unsigned __int64 v20; // rbx
  unsigned __int64 v21; // rdi

  if ( !(_BYTE)qword_5031648 )
  {
    v20 = *(_QWORD *)(a1 + 136);
    while ( v20 )
    {
      sub_30FAE30(*(_QWORD *)(v20 + 24));
      v21 = v20;
      v20 = *(_QWORD *)(v20 + 16);
      j_j___libc_free_0(v21);
    }
    *(_QWORD *)(a1 + 136) = 0;
    *(_QWORD *)(a1 + 144) = a1 + 128;
    *(_QWORD *)(a1 + 152) = a1 + 128;
    *(_QWORD *)(a1 + 160) = 0;
  }
  if ( a2 && !*(_BYTE *)(a1 + 360) )
  {
    v7 = *(_BYTE *)(a1 + 284) == 0;
    v8 = *(_QWORD **)(a1 + 264);
    *(_QWORD *)(a1 + 192) = 0;
    if ( v7 )
      v9 = *(unsigned int *)(a1 + 272);
    else
      v9 = *(unsigned int *)(a1 + 276);
    v10 = &v8[v9];
    if ( v8 != v10 )
    {
      while ( 1 )
      {
        v11 = *v8;
        v12 = v8;
        if ( *v8 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v10 == ++v8 )
          goto LABEL_10;
      }
      while ( v10 != v12 )
      {
        *(_QWORD *)(a1 + 192) += sub_30FCC90(a1, *(_QWORD *)(v11 + 8));
        v19 = v12 + 1;
        if ( v12 + 1 == v10 )
          break;
        while ( 1 )
        {
          v11 = *v19;
          v12 = v19;
          if ( *v19 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v10 == ++v19 )
            goto LABEL_10;
        }
      }
    }
LABEL_10:
    v13 = *(__int64 **)(a2 + 8);
    v14 = *(unsigned int *)(a2 + 16);
    v15 = &v13[v14];
    v16 = v13;
    if ( v13 != v15 )
    {
      v17 = *v13;
      if ( !*(_BYTE *)(a1 + 284) )
        goto LABEL_18;
LABEL_12:
      v18 = *(_QWORD **)(a1 + 264);
      a4 = *(unsigned int *)(a1 + 276);
      v14 = (__int64)&v18[a4];
      if ( v18 == (_QWORD *)v14 )
      {
LABEL_26:
        if ( (unsigned int)a4 >= *(_DWORD *)(a1 + 272) )
          goto LABEL_18;
        *(_DWORD *)(a1 + 276) = a4 + 1;
        *(_QWORD *)v14 = v17;
        ++*(_QWORD *)(a1 + 256);
LABEL_19:
        ++v16;
        *(_QWORD *)(a1 + 192) += sub_30FCC90(a1, *(_QWORD *)(v17 + 8));
        if ( v15 != v16 )
          goto LABEL_17;
      }
      else
      {
        while ( v17 != *v18 )
        {
          if ( (_QWORD *)v14 == ++v18 )
            goto LABEL_26;
        }
        while ( v15 != ++v16 )
        {
LABEL_17:
          v17 = *v16;
          if ( *(_BYTE *)(a1 + 284) )
            goto LABEL_12;
LABEL_18:
          sub_C8CC70(a1 + 256, v17, v14, a4, a5, a6);
          if ( (_BYTE)v14 )
            goto LABEL_19;
        }
      }
    }
  }
}
