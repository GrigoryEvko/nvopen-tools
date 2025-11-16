// Function: sub_27A3120
// Address: 0x27a3120
//
__int64 __fastcall sub_27A3120(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4)
{
  __int64 *v6; // r14
  unsigned int v7; // r13d
  _QWORD *v8; // r15
  __int64 v9; // rax
  int v10; // esi
  __int64 v11; // rdi
  int v12; // esi
  unsigned int v13; // edx
  _QWORD *v14; // rax
  _QWORD *v15; // r10
  __int64 v16; // r9
  __int64 v17; // rcx
  __int64 v18; // r8
  int v20; // eax
  int v21; // r9d
  __int64 v22; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+18h] [rbp-38h]

  v24 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( v24 != *(_QWORD *)a2 )
  {
    v6 = *(__int64 **)a2;
    v7 = 0;
    while ( 1 )
    {
      v8 = (_QWORD *)*v6;
      if ( (unsigned __int8 *)*v6 != a3 )
        break;
LABEL_10:
      if ( (__int64 *)v24 == ++v6 )
        return v7;
    }
    ++v7;
    sub_27A3020(a1, *v6, a3);
    if ( !a4 )
    {
LABEL_9:
      sub_F57030(a3, (__int64)v8, 1);
      sub_B45560(a3, (unsigned __int64)v8);
      sub_BD84D0((__int64)v8, (__int64)a3);
      sub_1031600(*(_QWORD *)(a1 + 240), (__int64)v8);
      sub_B43D60(v8);
      goto LABEL_10;
    }
    v9 = *(_QWORD *)(a1 + 248);
    v10 = *(_DWORD *)(v9 + 56);
    v11 = *(_QWORD *)(v9 + 40);
    if ( v10 )
    {
      v12 = v10 - 1;
      v13 = v12 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v14 = (_QWORD *)(v11 + 16LL * v13);
      v15 = (_QWORD *)*v14;
      if ( v8 == (_QWORD *)*v14 )
      {
LABEL_7:
        v16 = v14[1];
LABEL_8:
        v22 = v16;
        sub_BD84D0(v16, a4);
        sub_D6E4B0(*(_QWORD **)(a1 + 256), v22, 0, v17, v18, v22);
        goto LABEL_9;
      }
      v20 = 1;
      while ( v15 != (_QWORD *)-4096LL )
      {
        v21 = v20 + 1;
        v13 = v12 & (v20 + v13);
        v14 = (_QWORD *)(v11 + 16LL * v13);
        v15 = (_QWORD *)*v14;
        if ( v8 == (_QWORD *)*v14 )
          goto LABEL_7;
        v20 = v21;
      }
    }
    v16 = 0;
    goto LABEL_8;
  }
  return 0;
}
