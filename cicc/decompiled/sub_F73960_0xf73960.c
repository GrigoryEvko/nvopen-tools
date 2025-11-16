// Function: sub_F73960
// Address: 0xf73960
//
__int64 __fastcall sub_F73960(__int64 a1, unsigned __int64 *a2)
{
  __int64 result; // rax
  _QWORD *v3; // r12
  __int64 v4; // r14
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  _QWORD *v11; // r12
  _QWORD *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax

  result = *(unsigned int *)(a1 + 8);
  v3 = *(_QWORD **)a1;
  v4 = *(_QWORD *)a1 + 112 * result;
  if ( *(_QWORD *)a1 != v4 )
  {
    do
    {
      if ( a2 )
      {
        *a2 = 6;
        a2[1] = 0;
        v6 = v3[2];
        a2[2] = v6;
        if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
          sub_BD6050(a2, *v3 & 0xFFFFFFFFFFFFFFF8LL);
        a2[3] = 6;
        a2[4] = 0;
        v7 = v3[5];
        a2[5] = v7;
        if ( v7 != -4096 && v7 != 0 && v7 != -8192 )
          sub_BD6050(a2 + 3, v3[3] & 0xFFFFFFFFFFFFFFF8LL);
        v8 = v3[6];
        a2[7] = 6;
        a2[8] = 0;
        a2[6] = v8;
        v9 = v3[9];
        a2[9] = v9;
        if ( v9 != -4096 && v9 != 0 && v9 != -8192 )
          sub_BD6050(a2 + 7, v3[7] & 0xFFFFFFFFFFFFFFF8LL);
        a2[10] = 6;
        a2[11] = 0;
        v10 = v3[12];
        a2[12] = v10;
        if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
          sub_BD6050(a2 + 10, v3[10] & 0xFFFFFFFFFFFFFFF8LL);
        a2[13] = v3[13];
      }
      v3 += 14;
      a2 += 14;
    }
    while ( (_QWORD *)v4 != v3 );
    result = *(unsigned int *)(a1 + 8);
    v11 = *(_QWORD **)a1;
    v12 = (_QWORD *)(*(_QWORD *)a1 + 112 * result);
    if ( *(_QWORD **)a1 != v12 )
    {
      do
      {
        v13 = *(v12 - 2);
        v12 -= 14;
        if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
          sub_BD60C0(v12 + 10);
        v14 = v12[9];
        if ( v14 != -4096 && v14 != 0 && v14 != -8192 )
          sub_BD60C0(v12 + 7);
        v15 = v12[5];
        if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
          sub_BD60C0(v12 + 3);
        result = v12[2];
        if ( result != 0 && result != -4096 && result != -8192 )
          result = sub_BD60C0(v12);
      }
      while ( v12 != v11 );
    }
  }
  return result;
}
