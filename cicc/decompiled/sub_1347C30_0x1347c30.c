// Function: sub_1347C30
// Address: 0x1347c30
//
__int64 __fastcall sub_1347C30(_BYTE *a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5, __int64 a6, bool *a7)
{
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  unsigned __int64 v14; // rsi
  __int64 v15; // r8
  unsigned __int64 v16; // rax
  unsigned __int64 *v17; // r15
  __int64 v18; // r8
  bool v19; // al
  unsigned __int64 *v21; // rax
  __int64 v22; // r15
  __int64 v23; // rdi
  unsigned __int64 *v24; // r14
  pthread_mutex_t *mutex; // [rsp+8h] [rbp-68h]
  __int64 *v28; // [rsp+20h] [rbp-50h]
  __int64 v31; // [rsp+38h] [rbp-38h]

  mutex = (pthread_mutex_t *)(a2 + 128);
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 128)) )
  {
    sub_130AD90(a2 + 64);
    *(_BYTE *)(a2 + 168) = 1;
  }
  ++*(_QWORD *)(a2 + 120);
  if ( a1 != *(_BYTE **)(a2 + 112) )
  {
    ++*(_QWORD *)(a2 + 104);
    *(_QWORD *)(a2 + 112) = a1;
  }
  v31 = 0;
  v28 = (__int64 *)(a2 + 296);
  if ( a5 )
  {
    while ( 1 )
    {
      v17 = sub_1340BA0(a1, (__int64)v28);
      if ( !v17 )
        goto LABEL_15;
      v9 = sub_134BE50(a2 + 320, a3);
      v10 = v9;
      if ( !v9 )
        break;
      sub_134BCA0(a2 + 320, v9);
      if ( !*(_QWORD *)(v10 + 104) )
      {
        v11 = *(_QWORD *)(a2 + 5600);
        *(_QWORD *)(a2 + 5600) = v11 + 1;
        *(_QWORD *)(v10 + 8) = v11;
      }
      v12 = sub_1349810(v10, a3);
      v13 = *(unsigned int *)(a2 + 5608);
      v14 = *(_QWORD *)(v10 + 8);
      v15 = v12;
      v16 = *v17;
      v17[3] = v10;
      v17[1] = v15;
      v17[4] = v14;
      v17[2] = a3 | v17[2] & 0xFFF;
      *v17 = v13 & 0xFFFFEFFFF0000FFFLL | v16 & 0xFFFFEFFFF0000000LL | 0xE806000;
      if ( (unsigned __int8)sub_1341BA0((__int64)a1, *(_QWORD *)(a2 + 5616), v17, 0xE8u, 0) )
      {
        v21 = v17;
        v22 = v10;
        v23 = v10;
        v24 = v21;
        sub_1349DA0(v23, v21[1], v21[2] & 0xFFFFFFFFFFFFF000LL);
        sub_134BD00(a2 + 320, v22);
        sub_1340D30((__int64)a1, v28, v24);
LABEL_15:
        *a4 = 1;
        goto LABEL_16;
      }
      sub_1347380(a2, v10);
      sub_134BD00(a2 + 320, v10);
      v17[5] = (unsigned __int64)v17;
      v17[6] = (unsigned __int64)v17;
      if ( *(_QWORD *)a6 )
      {
        v17[5] = *(_QWORD *)(*(_QWORD *)a6 + 48LL);
        *(_QWORD *)(*(_QWORD *)a6 + 48LL) = v17;
        v17[6] = *(_QWORD *)(v17[6] + 40);
        *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a6 + 48LL) + 40LL) = *(_QWORD *)a6;
        *(_QWORD *)(v17[6] + 40) = v17;
        v17 = (unsigned __int64 *)v17[5];
      }
      ++v31;
      *(_QWORD *)a6 = v17;
      if ( a5 == v31 )
        goto LABEL_16;
    }
    sub_1340D30((__int64)a1, v28, v17);
  }
LABEL_16:
  sub_1347550((__int64)a1, a2, 0);
  v18 = sub_134BFA0(a2 + 320);
  v19 = 1;
  if ( !v18 )
    v19 = sub_1347320(a2);
  *a7 = v19;
  *(_BYTE *)(a2 + 168) = 0;
  pthread_mutex_unlock(mutex);
  return v31;
}
