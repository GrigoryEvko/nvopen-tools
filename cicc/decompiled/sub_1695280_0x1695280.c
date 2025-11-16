// Function: sub_1695280
// Address: 0x1695280
//
void __fastcall sub_1695280(__int64 **a1, __int64 a2, __int64 a3, unsigned int a4, unsigned int a5, int a6)
{
  __int64 v9; // rcx
  _QWORD *v10; // rax
  __int64 v11; // r15
  __int64 *v12; // rax
  __int64 *v13; // r14
  __int64 *v14; // rsi
  _QWORD *v15; // rdi
  _QWORD *v16; // rdi
  _QWORD *v17; // rcx
  unsigned __int64 v18; // r8
  unsigned int v19; // esi
  unsigned __int64 v20; // rax
  __int64 v21; // r11
  __int64 *v22; // rdx
  unsigned __int64 v23; // rdx
  __int64 v26; // [rsp+10h] [rbp-40h]

  v9 = 24LL * a5;
  v10 = *(_QWORD **)(a3 + 24);
  if ( v10 )
  {
    if ( a4 )
    {
      v11 = *(_QWORD *)(v10[3] + v9 + 16);
      if ( !(_DWORD)v11 )
        return;
    }
    else
    {
      v11 = *(_QWORD *)(*v10 + v9 + 16);
      if ( !(_DWORD)v11 )
        return;
    }
  }
  else
  {
    v11 = *(_QWORD *)(24LL * a5 + 0x10);
    if ( !(_DWORD)v11 )
      return;
  }
  v26 = 3LL * a5;
  v12 = (__int64 *)sub_2207820(16LL * (unsigned int)v11);
  v13 = v12;
  if ( v12 )
  {
    v14 = &v12[2 * (unsigned int)v11];
    do
    {
      *v12 = 0;
      v12 += 2;
      *(v12 - 1) = 0;
    }
    while ( v14 != v12 );
  }
  v15 = *(_QWORD **)(a3 + 24);
  if ( v15 )
  {
    if ( a4 )
      v15 = (_QWORD *)v15[3];
    else
      v15 = (_QWORD *)*v15;
  }
  v16 = &v15[v26];
  v17 = (_QWORD *)*v16;
  if ( (_QWORD *)*v16 != v16 )
  {
    v18 = v17[3];
    v19 = 0;
    *v13 = v17[2];
    v13[1] = v18;
    while ( 1 )
    {
      v17 = (_QWORD *)*v17;
      ++v19;
      if ( v16 == v17 )
        break;
      v20 = v17[3];
      v21 = v17[2];
      v22 = &v13[2 * v19];
      v22[1] = v20;
      *v22 = v21;
      v23 = v20 + v18;
      if ( v20 < v18 )
        v20 = v18;
      if ( v23 < v20 )
        v23 = -1;
      v18 = v23;
    }
    sub_1694FA0(a1, a2, v13, (unsigned int)v11, v18, a4, a6);
    goto LABEL_21;
  }
  sub_1694FA0(a1, a2, v13, (unsigned int)v11, 0, a4, a6);
  if ( v13 )
LABEL_21:
    j_j___libc_free_0_0(v13);
}
