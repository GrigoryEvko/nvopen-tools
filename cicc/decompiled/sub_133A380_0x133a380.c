// Function: sub_133A380
// Address: 0x133a380
//
__int64 __fastcall sub_133A380(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v10; // rbx
  unsigned int v11; // r12d
  unsigned __int64 v12; // rbx
  __int64 v13; // r12
  _QWORD *v14; // rax
  __int64 v15; // r15
  _QWORD *v16; // r14
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx

  if ( pthread_mutex_trylock(&stru_4F96C00) )
  {
    sub_130AD90((__int64)&xmmword_4F96BC0);
    byte_4F96C28 = 1;
  }
  ++*((_QWORD *)&xmmword_4F96BF0 + 1);
  if ( a1 != (_QWORD)xmmword_4F96BF0 )
  {
    ++qword_4F96BE8;
    *(_QWORD *)&xmmword_4F96BF0 = a1;
  }
  v10 = a7 | a6 | a5;
  v11 = 1;
  if ( !(v10 | a4) )
  {
    v12 = *(_QWORD *)(a2 + 8);
    if ( v12 > 0xFFFFFFFF
      || (v13 = qword_50579C0[v12]) == 0
      || *(_DWORD *)(v13 + 78928) < dword_5057900[0]
      || (unsigned int)sub_1317080(qword_50579C0[v12], 0)
      || (unsigned int)sub_1317080(v13, 1u) )
    {
      v11 = 14;
    }
    else
    {
      sub_133A200(a1, v12);
      sub_1315300(a1, v13);
      sub_1315160(a1, v13, 0, 1u);
      v14 = sub_1322320(4097);
      *((_BYTE *)v14 + 4) = 1;
      v15 = (__int64)v14;
      v16 = sub_1322320(v12);
      sub_131DCA0((__int64)v16);
      sub_131DE10(a1, (__int64)v16, v13);
      sub_13226C0(v15, (__int64)v16, 1);
      sub_13158D0(a1, v13);
      v17 = sub_1322320(v12);
      v18 = qword_4F96BA0;
      *((_BYTE *)v17 + 4) = 0;
      v17[1] = v17;
      v17[2] = v17;
      v19 = *(_QWORD *)(v18 + 16);
      if ( v19 )
      {
        v17[1] = *(_QWORD *)(v19 + 16);
        *(_QWORD *)(*(_QWORD *)(v18 + 16) + 16LL) = v17;
        v17[2] = *(_QWORD *)(v17[2] + 8LL);
        *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v18 + 16) + 16LL) + 8LL) = *(_QWORD *)(v18 + 16);
        *(_QWORD *)(v17[2] + 8LL) = v17;
        v17 = (_QWORD *)v17[1];
      }
      *(_QWORD *)(v18 + 16) = v17;
      v11 = 0;
      sub_1339EA0(a1, v12);
    }
  }
  byte_4F96C28 = 0;
  pthread_mutex_unlock(&stru_4F96C00);
  return v11;
}
