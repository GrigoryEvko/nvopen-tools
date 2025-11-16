// Function: sub_302AC40
// Address: 0x302ac40
//
void __fastcall sub_302AC40(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  _QWORD *v3; // rax
  _QWORD *v4; // r8
  _QWORD *v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rdx
  _BYTE **v10; // rbx
  _BYTE **v11; // r14
  __int64 v12; // r13
  __m128i si128; // xmm0
  __m128i *v14; // rdx
  _BYTE *v15; // r10
  _BYTE *v16; // [rsp-40h] [rbp-40h]

  v3 = *(_QWORD **)(a1 + 1160);
  if ( v3 )
  {
    v4 = (_QWORD *)(a1 + 1152);
    v7 = (_QWORD *)(a1 + 1152);
    do
    {
      while ( 1 )
      {
        v8 = v3[2];
        v9 = v3[3];
        if ( v3[4] >= a2 )
          break;
        v3 = (_QWORD *)v3[3];
        if ( !v9 )
          goto LABEL_6;
      }
      v7 = v3;
      v3 = (_QWORD *)v3[2];
    }
    while ( v8 );
LABEL_6:
    if ( v4 != v7 && v7[4] <= a2 )
    {
      v10 = (_BYTE **)v7[5];
      v11 = (_BYTE **)v7[6];
      v12 = *(_QWORD *)(a1 + 200) + 1288LL;
      while ( v11 != v10 )
      {
        v14 = *(__m128i **)(a3 + 32);
        v15 = *v10;
        if ( *(_QWORD *)(a3 + 24) - (_QWORD)v14 > 0x15u )
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_4327150);
          v14[1].m128i_i32[0] = 1701601889;
          v14[1].m128i_i16[2] = 2314;
          *v14 = si128;
          *(_QWORD *)(a3 + 32) += 22LL;
        }
        else
        {
          v16 = *v10;
          sub_CB6200(a3, "\t// demoted variable\n\t", 0x16u);
          v15 = v16;
        }
        ++v10;
        sub_3029BF0(a1, v15, a3, 1, v12);
      }
    }
  }
}
