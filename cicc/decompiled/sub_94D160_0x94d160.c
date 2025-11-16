// Function: sub_94D160
// Address: 0x94d160
//
__int64 __fastcall sub_94D160(__int64 a1, __int64 a2, unsigned int a3, int a4, char a5, __int64 a6)
{
  unsigned int v10; // ebx
  unsigned __int64 *v11; // r8
  __int64 v12; // rax
  __int64 *v13; // rdi
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rax
  __m128i *v17; // rax
  unsigned __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v21; // r14
  unsigned __int64 v22; // rsi
  __int64 v23; // [rsp+0h] [rbp-A0h]
  __int64 v24; // [rsp+8h] [rbp-98h]
  __int64 v25; // [rsp+10h] [rbp-90h] BYREF
  __int64 v26; // [rsp+20h] [rbp-80h] BYREF
  __m128i *v27; // [rsp+28h] [rbp-78h]
  __m128i *v28; // [rsp+30h] [rbp-70h]
  __m128i *v29; // [rsp+38h] [rbp-68h]
  _BYTE v30[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v31; // [rsp+60h] [rbp-40h]

  v10 = (unsigned __int8)a4 << 16;
  LOBYTE(v10) = (16 * a5) | 5;
  v11 = *(unsigned __int64 **)(a6 + 16);
  v23 = (__int64)v11;
  v24 = v11[2];
  v12 = sub_91A390(*(_QWORD *)(a2 + 32) + 8LL, *v11, 0, *(_QWORD *)(a2 + 32));
  v13 = *(__int64 **)(a2 + 32);
  v25 = v12;
  v14 = sub_90A810(v13, a3, (__int64)&v25, 1u);
  v15 = sub_BCB2D0(*(_QWORD *)(a2 + 40));
  v16 = sub_ACD640(v15, v10, 0);
  if ( a4 == 14 )
  {
    v21 = *(_QWORD *)(v24 + 16);
    v26 = v16;
    v27 = sub_92F410(a2, v23);
    v28 = sub_92F410(a2, v24);
    v22 = 0;
    v29 = sub_92F410(a2, v21);
    v31 = 257;
    if ( v14 )
      v22 = *(_QWORD *)(v14 + 24);
    v19 = sub_921880((unsigned int **)(a2 + 48), v22, v14, (int)&v26, 4, (__int64)v30, 0);
  }
  else
  {
    v26 = v16;
    v27 = sub_92F410(a2, v23);
    v17 = sub_92F410(a2, v24);
    v31 = 257;
    v18 = 0;
    v28 = v17;
    if ( v14 )
      v18 = *(_QWORD *)(v14 + 24);
    v19 = sub_921880((unsigned int **)(a2 + 48), v18, v14, (int)&v26, 3, (__int64)v30, 0);
  }
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = v19;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
