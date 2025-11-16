// Function: sub_1316790
// Address: 0x1316790
//
__int64 __fastcall sub_1316790(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, unsigned int a6)
{
  unsigned __int64 *v6; // r12
  unsigned int v7; // ebx
  __int64 v8; // rax
  pthread_mutex_t *v9; // r15
  _QWORD *v10; // r14
  __int64 v11; // r13
  __int64 v12; // rax
  _QWORD *v13; // r13
  __int64 v14; // r14
  _QWORD *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  size_t v20; // rdx
  __int64 v21; // rax
  void *v22; // rcx
  __int64 result; // rax
  __int64 v24; // rax
  int v25; // eax
  __int64 v26; // [rsp+8h] [rbp-88h]
  __int64 v27; // [rsp+10h] [rbp-80h]
  _BYTE *v29; // [rsp+20h] [rbp-70h]
  __int64 *v30; // [rsp+28h] [rbp-68h]
  char v32; // [rsp+38h] [rbp-58h]
  int v33; // [rsp+38h] [rbp-58h]
  int v36; // [rsp+4Ch] [rbp-44h]
  unsigned int v37[13]; // [rsp+5Ch] [rbp-34h] BYREF

  v6 = 0;
  v7 = 0;
  v30 = (__int64 *)((char *)&unk_5260DE0 + 40 * a5);
  v26 = *(_QWORD *)a3 + (unsigned __int16)(*(_WORD *)(a3 + 20) - *(_QWORD *)a3);
  v27 = (unsigned __int16)a6;
  v8 = sub_1315920(a1, a2, a5, v37);
  v32 = 1;
  v9 = (pthread_mutex_t *)(v8 + 64);
  v29 = (_BYTE *)(v8 + 104);
  v10 = (_QWORD *)v8;
  v11 = v8 + 200;
  if ( pthread_mutex_trylock((pthread_mutex_t *)(v8 + 64)) )
  {
LABEL_27:
    sub_130AD90((__int64)v10);
    *v29 = 1;
  }
  while ( 2 )
  {
    ++v10[7];
    if ( a1 != v10[6] )
    {
      ++v10[5];
      v10[6] = a1;
    }
    if ( v7 >= a6 )
      break;
    v12 = v11;
    v13 = v10;
    v14 = v12;
    while ( 1 )
    {
      v15 = (_QWORD *)v13[24];
      if ( v15 )
      {
        v16 = (*v15 >> 28) & 0x3FFLL;
        if ( ((*v15 >> 28) & 0x3FF) != 0 )
        {
          if ( a6 - v7 <= (unsigned int)v16 )
            LODWORD(v16) = a6 - v7;
          v33 = v16;
          sub_1314190(v15, v30, v16, v26 + 8 * (v7 - v27));
          v25 = v33;
          v32 = 1;
          v7 += v25;
          goto LABEL_15;
        }
        if ( *(_DWORD *)(a2 + 78928) >= dword_5057900[0] )
        {
          v15[5] = v15;
          v15[6] = v15;
          v17 = v13[27];
          if ( v17 )
          {
            v15[5] = *(_QWORD *)(v17 + 48);
            *(_QWORD *)(v13[27] + 48LL) = v15;
            v15[6] = *(_QWORD *)(v15[6] + 40LL);
            *(_QWORD *)(*(_QWORD *)(v13[27] + 48LL) + 40LL) = v13[27];
            *(_QWORD *)(v15[6] + 40LL) = v15;
            v15 = (_QWORD *)v15[5];
          }
          v13[27] = v15;
        }
      }
      v18 = sub_133FAA0(v14);
      if ( v18 )
      {
        ++v13[21];
        --v13[23];
        v13[24] = v18;
        goto LABEL_15;
      }
      v13[24] = 0;
      if ( !v6 )
        break;
      ++v13[20];
      ++v13[22];
      v13[24] = v6;
      v6 = 0;
LABEL_15:
      if ( a6 <= v7 )
        goto LABEL_16;
    }
    v24 = v14;
    v10 = v13;
    v11 = v24;
    if ( v32 )
    {
      *v29 = 0;
      pthread_mutex_unlock(v9);
      v32 = 0;
      v6 = sub_1316650(a1, a2, a5, v37[0], (__int64)v30);
      if ( !pthread_mutex_trylock(v9) )
        continue;
      goto LABEL_27;
    }
    break;
  }
  v13 = v10;
LABEL_16:
  v13[14] += v7;
  v19 = *(_QWORD *)(a3 + 8);
  v13[17] += v7;
  v13[16] += v19;
  ++v13[18];
  *(_QWORD *)(a3 + 8) = 0;
  *((_BYTE *)v13 + 104) = 0;
  pthread_mutex_unlock(v9);
  if ( v6 )
    sub_13152A0(a1, a2, v6);
  v20 = 8LL * (unsigned __int16)v7;
  v21 = (unsigned __int16)(*(_WORD *)(a3 + 20) - *(_QWORD *)a3);
  v22 = (void *)(*(_QWORD *)a3 + v21 - v20);
  if ( (unsigned __int16)a6 > (unsigned __int16)v7 )
    v22 = memmove((void *)(*(_QWORD *)a3 + v21 - v20), (const void *)(*(_QWORD *)a3 + v21 - 8 * v27), v20);
  result = a3;
  *(_QWORD *)a3 = v22;
  if ( a1 )
  {
    v36 = *(_DWORD *)(a1 + 152);
    result = (unsigned int)(v36 - 1);
    *(_DWORD *)(a1 + 152) = result;
    if ( v36 - 1 < 0 )
    {
      result = sub_1314130((_DWORD *)(a1 + 152), (unsigned __int64 *)(a1 + 112));
      if ( (_BYTE)result )
        return sub_1315160(a1, a2, 0, 0);
    }
  }
  return result;
}
