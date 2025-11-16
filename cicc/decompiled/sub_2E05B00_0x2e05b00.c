// Function: sub_2E05B00
// Address: 0x2e05b00
//
void __fastcall sub_2E05B00(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rax
  unsigned __int64 v6; // r13
  char v7; // al
  __int64 v8; // r15
  __int64 *v9; // rbx
  __int64 v10; // rax
  __int64 *v11; // r13
  __int64 v12; // rax
  __int64 v13; // r12
  _QWORD *v14; // r14
  unsigned __int64 *v15; // rdi
  unsigned __int64 v16; // rcx
  __int64 *i; // [rsp-48h] [rbp-48h]

  if ( (_BYTE)qword_501E9A8 )
  {
    v4 = a2;
    if ( sub_B92180(*(_QWORD *)a2) )
    {
      v5 = sub_22077B0(0x498u);
      if ( v5 )
      {
        *(_QWORD *)v5 = 0;
        *(_QWORD *)(v5 + 24) = v5 + 40;
        *(_QWORD *)(v5 + 32) = 0x400000000LL;
        *(_QWORD *)(v5 + 72) = v5 + 88;
        *(_QWORD *)(v5 + 152) = v5 + 136;
        *(_QWORD *)(v5 + 160) = v5 + 136;
        *(_QWORD *)(v5 + 208) = v5 + 224;
        *(_QWORD *)(v5 + 216) = 0x2000000000LL;
        *(_WORD *)(v5 + 992) = 0;
        *(_QWORD *)(v5 + 1000) = v5 + 1016;
        *(_QWORD *)(v5 + 1008) = 0x800000000LL;
        *(_QWORD *)(v5 + 1080) = v5 + 1096;
        *(_QWORD *)(v5 + 8) = 0;
        *(_QWORD *)(v5 + 16) = 0;
        *(_QWORD *)(v5 + 80) = 0;
        *(_QWORD *)(v5 + 88) = 0;
        *(_QWORD *)(v5 + 96) = 1;
        *(_QWORD *)(v5 + 104) = 0;
        *(_QWORD *)(v5 + 112) = a3;
        *(_DWORD *)(v5 + 136) = 0;
        *(_QWORD *)(v5 + 144) = 0;
        *(_QWORD *)(v5 + 168) = 0;
        *(_QWORD *)(v5 + 176) = 0;
        *(_QWORD *)(v5 + 184) = 0;
        *(_QWORD *)(v5 + 192) = 0;
        *(_DWORD *)(v5 + 200) = 0;
        *(_QWORD *)(v5 + 1088) = 0x200000000LL;
        *(_QWORD *)(v5 + 1112) = 0;
        *(_QWORD *)(v5 + 1120) = 0;
        *(_QWORD *)(v5 + 1128) = 0;
        *(_DWORD *)(v5 + 1136) = 0;
        *(_QWORD *)(v5 + 1144) = 0;
        *(_QWORD *)(v5 + 1152) = 0;
        *(_QWORD *)(v5 + 1160) = 0;
        *(_DWORD *)(v5 + 1168) = 0;
      }
      v6 = *a1;
      *a1 = v5;
      if ( v6 )
      {
        sub_2DFA2C0(v6);
        a2 = 1176;
        j_j___libc_free_0(v6);
      }
      v7 = sub_2E799E0(v4, a2);
      sub_2E05630(*a1, v4, v7);
    }
    else
    {
      for ( i = *(__int64 **)(a2 + 328); (__int64 *)(a2 + 320) != i; i = (__int64 *)i[1] )
      {
        v8 = i[7];
        v9 = i + 5;
        if ( (__int64 *)v8 != i + 6 )
        {
          do
          {
            if ( !v8 )
              BUG();
            v10 = v8;
            if ( ((*(__int64 *)v8 >> 2) & 1) == 0 && (*(_BYTE *)(v8 + 44) & 8) != 0 )
            {
              do
                v10 = *(_QWORD *)(v10 + 8);
              while ( (*(_BYTE *)(v10 + 44) & 8) != 0 );
            }
            v11 = *(__int64 **)(v10 + 8);
            if ( (unsigned __int16)(*(_WORD *)(v8 + 68) - 14) <= 4u )
            {
              v12 = v8;
              if ( ((*(__int64 *)v8 >> 2) & 1) == 0 && (*(_BYTE *)(v8 + 44) & 8) != 0 )
              {
                do
                  v12 = *(_QWORD *)(v12 + 8);
                while ( (*(_BYTE *)(v12 + 44) & 8) != 0 );
              }
              v13 = *(_QWORD *)(v12 + 8);
              while ( v8 != v13 )
              {
                v14 = (_QWORD *)v8;
                v8 = *(_QWORD *)(v8 + 8);
                sub_2E31080(v9, v14);
                v15 = (unsigned __int64 *)v14[1];
                v16 = *v14 & 0xFFFFFFFFFFFFFFF8LL;
                *v15 = v16 | *v15 & 7;
                *(_QWORD *)(v16 + 8) = v15;
                *v14 &= 7uLL;
                v14[1] = 0;
                sub_2E310F0(v9, v14);
              }
            }
            v8 = (__int64)v11;
          }
          while ( v11 != i + 6 );
        }
      }
    }
  }
}
