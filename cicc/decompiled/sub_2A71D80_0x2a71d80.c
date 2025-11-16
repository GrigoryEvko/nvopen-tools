// Function: sub_2A71D80
// Address: 0x2a71d80
//
void __fastcall sub_2A71D80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 i; // r13
  unsigned __int8 *v10; // rsi
  int v11; // eax
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // r8
  unsigned int v16; // edx
  __int64 v17; // rcx
  __int64 v18; // rdi
  _QWORD *v19; // rbx
  _QWORD *v20; // r12
  __int64 v21; // rax
  _BYTE *v22; // r15
  _BYTE *v23; // r15
  _BYTE *v24; // rbx
  unsigned __int8 *v25; // r13
  __int64 v26; // rsi
  _QWORD *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r15
  __int64 v30; // rsi
  _QWORD *v31; // rax
  __int64 v32; // rdx
  int v33; // ecx
  _BYTE *v34; // [rsp+10h] [rbp-50h] BYREF
  __int64 v35; // [rsp+18h] [rbp-48h]
  _BYTE v36[64]; // [rsp+20h] [rbp-40h] BYREF

  v8 = *(_QWORD *)(a2 + 16);
  if ( *(_BYTE *)a2 )
  {
    if ( !v8 )
      goto LABEL_8;
    while ( 1 )
    {
      v29 = *(_QWORD *)(v8 + 24);
      if ( *(_BYTE *)v29 > 0x1Cu )
      {
        v30 = *(_QWORD *)(v29 + 40);
        if ( *(_BYTE *)(a1 + 68) )
        {
          v31 = *(_QWORD **)(a1 + 48);
          v32 = (__int64)&v31[*(unsigned int *)(a1 + 60)];
          if ( v31 == (_QWORD *)v32 )
            goto LABEL_40;
          while ( v30 != *v31 )
          {
            if ( (_QWORD *)v32 == ++v31 )
              goto LABEL_40;
          }
        }
        else if ( !sub_C8CA60(a1 + 40, v30) )
        {
          goto LABEL_40;
        }
        sub_2A71C10((unsigned __int64 *)a1, (unsigned __int8 *)v29, v32, a4);
      }
LABEL_40:
      v8 = *(_QWORD *)(v8 + 8);
      if ( !v8 )
        goto LABEL_8;
    }
  }
  for ( i = 0x8000000000041LL; v8; v8 = *(_QWORD *)(v8 + 8) )
  {
    v10 = *(unsigned __int8 **)(v8 + 24);
    v11 = *v10;
    if ( (unsigned __int8)v11 > 0x1Cu )
    {
      v12 = (unsigned int)(v11 - 34);
      if ( (unsigned __int8)v12 <= 0x33u )
      {
        if ( _bittest64(&i, v12) )
          sub_2A70990(a1, v10);
      }
    }
  }
LABEL_8:
  v13 = *(unsigned int *)(a1 + 2592);
  v14 = *(_QWORD *)(a1 + 2576);
  if ( (_DWORD)v13 )
  {
    v15 = (unsigned int)(v13 - 1);
    v16 = v15 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v17 = v14 + 72LL * v16;
    v18 = *(_QWORD *)v17;
    if ( a2 == *(_QWORD *)v17 )
    {
LABEL_10:
      if ( v17 != v14 + 72 * v13 )
      {
        v19 = *(_QWORD **)(v17 + 40);
        v34 = v36;
        v35 = 0x200000000LL;
        v20 = &v19[*(unsigned int *)(v17 + 48)];
        if ( v20 != v19 )
        {
          v21 = 0;
          do
          {
            v22 = (_BYTE *)*v19;
            if ( *(_BYTE *)*v19 > 0x1Cu )
            {
              v17 = HIDWORD(v35);
              if ( v21 + 1 > (unsigned __int64)HIDWORD(v35) )
              {
                sub_C8D5F0((__int64)&v34, v36, v21 + 1, 8u, v15, a6);
                v21 = (unsigned int)v35;
              }
              *(_QWORD *)&v34[8 * v21] = v22;
              v21 = (unsigned int)(v35 + 1);
              LODWORD(v35) = v35 + 1;
            }
            ++v19;
          }
          while ( v20 != v19 );
          v23 = v34;
          v24 = &v34[8 * v21];
          if ( v24 != v34 )
          {
            while ( 1 )
            {
              v25 = *(unsigned __int8 **)v23;
              v26 = *(_QWORD *)(*(_QWORD *)v23 + 40LL);
              if ( *(_BYTE *)(a1 + 68) )
              {
                v27 = *(_QWORD **)(a1 + 48);
                v28 = (__int64)&v27[*(unsigned int *)(a1 + 60)];
                if ( v27 == (_QWORD *)v28 )
                  goto LABEL_25;
                while ( v26 != *v27 )
                {
                  if ( (_QWORD *)v28 == ++v27 )
                    goto LABEL_25;
                }
              }
              else if ( !sub_C8CA60(a1 + 40, v26) )
              {
                goto LABEL_25;
              }
              sub_2A71C10((unsigned __int64 *)a1, v25, v28, v17);
LABEL_25:
              v23 += 8;
              if ( v24 == v23 )
              {
                v23 = v34;
                break;
              }
            }
          }
          if ( v23 != v36 )
            _libc_free((unsigned __int64)v23);
        }
      }
    }
    else
    {
      v33 = 1;
      while ( v18 != -4096 )
      {
        a6 = (unsigned int)(v33 + 1);
        v16 = v15 & (v33 + v16);
        v17 = v14 + 72LL * v16;
        v18 = *(_QWORD *)v17;
        if ( a2 == *(_QWORD *)v17 )
          goto LABEL_10;
        v33 = a6;
      }
    }
  }
}
