// Function: sub_2299720
// Address: 0x2299720
//
__int64 __fastcall sub_2299720(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 result; // rax
  __int64 v5; // r15
  __int64 v6; // rdx
  __int64 j; // r12
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rbx
  __int64 v12; // r12
  __int64 i; // r13
  _BYTE *v14; // rbx
  _DWORD *v15; // rdx
  _BYTE *v16; // r10
  _BYTE *v17; // r10
  __int64 v18; // rdx
  _BYTE *v19; // rax
  void *v20; // rdx
  __int64 v21; // rdi
  unsigned int v22; // ebx
  __m128i *v23; // rdx
  __m128i si128; // xmm0
  __int64 v25; // rdi
  void *v26; // rdx
  __int64 v27; // r13
  unsigned __int64 v28; // rax
  _WORD *v29; // rdx
  __int64 v30; // rdx
  __int64 v33; // [rsp+10h] [rbp-80h]
  __int64 v35; // [rsp+28h] [rbp-68h]
  __int64 v36; // [rsp+28h] [rbp-68h]
  __int64 v37; // [rsp+30h] [rbp-60h]
  __int64 v38; // [rsp+38h] [rbp-58h]
  __int64 v39; // [rsp+40h] [rbp-50h]
  __int64 v40; // [rsp+48h] [rbp-48h]
  __int64 v41[7]; // [rsp+58h] [rbp-38h] BYREF

  result = *(_QWORD *)(a2 + 24);
  v5 = *(_QWORD *)(result + 80);
  v6 = result + 72;
  v40 = result + 72;
  if ( result + 72 != v5 )
  {
    if ( !v5 )
      BUG();
    while ( 1 )
    {
      j = *(_QWORD *)(v5 + 32);
      result = v5 + 24;
      if ( j != v5 + 24 )
        break;
      v5 = *(_QWORD *)(v5 + 8);
      if ( v6 == v5 )
        return result;
      if ( !v5 )
        BUG();
    }
    if ( v40 != v5 )
    {
      v8 = v5;
      do
      {
        v9 = 0;
        if ( j )
          v9 = j - 24;
        v39 = v9;
        v10 = v9;
        if ( (unsigned __int8)sub_B46420(v9) || (unsigned __int8)sub_B46490(v10) )
        {
          v11 = j;
          if ( v40 != v8 )
          {
            v38 = v8;
            v37 = j;
            v12 = v8;
            i = v11;
            do
            {
              v14 = (_BYTE *)(i - 24);
              if ( !i )
                v14 = 0;
              if ( (unsigned __int8)sub_B46420((__int64)v14) || (unsigned __int8)sub_B46490((__int64)v14) )
              {
                v15 = *(_DWORD **)(a1 + 32);
                if ( *(_QWORD *)(a1 + 24) - (_QWORD)v15 <= 3u )
                {
                  v16 = (_BYTE *)sub_CB6200(a1, "Src:", 4u);
                }
                else
                {
                  *v15 = 979595859;
                  v16 = (_BYTE *)a1;
                  *(_QWORD *)(a1 + 32) += 4LL;
                }
                v35 = (__int64)v16;
                sub_A69870(v39, v16, 0);
                v17 = (_BYTE *)v35;
                v18 = *(_QWORD *)(v35 + 32);
                if ( (unsigned __int64)(*(_QWORD *)(v35 + 24) - v18) <= 8 )
                {
                  v17 = (_BYTE *)sub_CB6200(v35, " --> Dst:", 9u);
                }
                else
                {
                  *(_BYTE *)(v18 + 8) = 58;
                  *(_QWORD *)v18 = 0x747344203E2D2D20LL;
                  *(_QWORD *)(v35 + 32) += 9LL;
                }
                v36 = (__int64)v17;
                sub_A69870((__int64)v14, v17, 0);
                v19 = *(_BYTE **)(v36 + 32);
                if ( *(_BYTE **)(v36 + 24) == v19 )
                {
                  sub_CB6200(v36, (unsigned __int8 *)"\n", 1u);
                }
                else
                {
                  *v19 = 10;
                  ++*(_QWORD *)(v36 + 32);
                }
                v20 = *(void **)(a1 + 32);
                if ( *(_QWORD *)(a1 + 24) - (_QWORD)v20 <= 0xEu )
                {
                  sub_CB6200(a1, "  da analyze - ", 0xFu);
                }
                else
                {
                  qmemcpy(v20, "  da analyze - ", 15);
                  *(_QWORD *)(a1 + 32) += 15LL;
                }
                sub_2297CA0(v41, a2, v39, v14);
                v21 = v41[0];
                if ( v41[0] )
                {
                  if ( a4 )
                  {
                    if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v41[0] + 72LL))(v41[0], a3) )
                      sub_904010(a1, "normalized - ");
                    v21 = v41[0];
                  }
                  v22 = 1;
                  sub_228CF00(v21, a1);
                  v33 = i;
                  while ( (*(unsigned int (__fastcall **)(__int64))(*(_QWORD *)v41[0] + 40LL))(v41[0]) >= v22 )
                  {
                    if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v41[0] + 96LL))(v41[0], v22) )
                    {
                      v23 = *(__m128i **)(a1 + 32);
                      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v23 <= 0x1Cu )
                      {
                        v25 = sub_CB6200(a1, "  da analyze - split level = ", 0x1Du);
                      }
                      else
                      {
                        si128 = _mm_load_si128((const __m128i *)&xmmword_43660F0);
                        v25 = a1;
                        qmemcpy(&v23[1], "plit level = ", 13);
                        *v23 = si128;
                        *(_QWORD *)(a1 + 32) += 29LL;
                      }
                      sub_CB59D0(v25, v22);
                      v26 = *(void **)(a1 + 32);
                      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v26 <= 0xDu )
                      {
                        v27 = sub_CB6200(a1, ", iteration = ", 0xEu);
                      }
                      else
                      {
                        v27 = a1;
                        qmemcpy(v26, ", iteration = ", 14);
                        *(_QWORD *)(a1 + 32) += 14LL;
                      }
                      v28 = sub_2296310(a2, v41[0], v22);
                      sub_D955C0(v28, v27);
                      v29 = *(_WORD **)(a1 + 32);
                      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v29 <= 1u )
                      {
                        sub_CB6200(a1, (unsigned __int8 *)"!\n", 2u);
                      }
                      else
                      {
                        *v29 = 2593;
                        *(_QWORD *)(a1 + 32) += 2LL;
                      }
                    }
                    ++v22;
                  }
                  i = v33;
                }
                else
                {
                  v30 = *(_QWORD *)(a1 + 32);
                  if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v30) <= 5 )
                  {
                    sub_CB6200(a1, (unsigned __int8 *)"none!\n", 6u);
                  }
                  else
                  {
                    *(_DWORD *)v30 = 1701736302;
                    *(_WORD *)(v30 + 4) = 2593;
                    *(_QWORD *)(a1 + 32) += 6LL;
                  }
                }
                if ( v41[0] )
                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v41[0] + 8LL))(v41[0]);
              }
              for ( i = *(_QWORD *)(i + 8); i == v12 - 24 + 48; i = *(_QWORD *)(v12 + 32) )
              {
                v12 = *(_QWORD *)(v12 + 8);
                if ( v40 == v12 )
                  goto LABEL_42;
                if ( !v12 )
                  BUG();
              }
            }
            while ( v40 != v12 );
LABEL_42:
            v8 = v38;
            j = v37;
          }
        }
        for ( j = *(_QWORD *)(j + 8); ; j = *(_QWORD *)(v8 + 32) )
        {
          result = v8 - 24 + 48;
          if ( j != result )
            break;
          v8 = *(_QWORD *)(v8 + 8);
          if ( v40 == v8 )
            return result;
          if ( !v8 )
            BUG();
        }
      }
      while ( v8 != v40 );
    }
  }
  return result;
}
