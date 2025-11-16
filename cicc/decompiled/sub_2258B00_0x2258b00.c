// Function: sub_2258B00
// Address: 0x2258b00
//
__int64 __fastcall sub_2258B00(__int64 a1, __int64 a2)
{
  __int64 *v4; // rdi
  __int64 result; // rax
  __int64 v6; // r10
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  size_t v13; // rdx
  __int64 v14; // r9
  __int64 v15; // r8
  size_t *v16; // r9
  size_t **v17; // r14
  __int64 v18; // [rsp+0h] [rbp-60h]
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h]
  __int64 v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h]
  size_t n; // [rsp+28h] [rbp-38h]

  v4 = (__int64 *)(a1 + 16);
  *((_DWORD *)v4 - 4) = *(_DWORD *)a2;
  *((_DWORD *)v4 - 2) = *(_DWORD *)(a2 + 8);
  *(_QWORD *)(a1 + 16) = a1 + 32;
  sub_2257AB0(v4, *(_BYTE **)(a2 + 16), *(_QWORD *)(a2 + 16) + *(_QWORD *)(a2 + 24));
  *(_QWORD *)(a1 + 48) = a1 + 64;
  sub_2257AB0((__int64 *)(a1 + 48), *(_BYTE **)(a2 + 48), *(_QWORD *)(a2 + 48) + *(_QWORD *)(a2 + 56));
  *(_QWORD *)(a1 + 80) = a1 + 96;
  sub_2257AB0((__int64 *)(a1 + 80), *(_BYTE **)(a2 + 80), *(_QWORD *)(a2 + 80) + *(_QWORD *)(a2 + 88));
  *(_QWORD *)(a1 + 112) = a1 + 128;
  sub_2257AB0((__int64 *)(a1 + 112), *(_BYTE **)(a2 + 112), *(_QWORD *)(a2 + 112) + *(_QWORD *)(a2 + 120));
  *(_QWORD *)(a1 + 144) = a1 + 160;
  sub_2257AB0((__int64 *)(a1 + 144), *(_BYTE **)(a2 + 144), *(_QWORD *)(a2 + 144) + *(_QWORD *)(a2 + 152));
  *(_QWORD *)(a1 + 176) = a1 + 192;
  sub_2257AB0((__int64 *)(a1 + 176), *(_BYTE **)(a2 + 176), *(_QWORD *)(a2 + 176) + *(_QWORD *)(a2 + 184));
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0x1000000000LL;
  if ( *(_DWORD *)(a2 + 220) )
  {
    sub_C92620(a1 + 208, *(_DWORD *)(a2 + 216));
    v6 = *(_QWORD *)(a1 + 208);
    v7 = *(_QWORD *)(a2 + 208);
    v8 = *(unsigned int *)(a1 + 216);
    v9 = 8 * v8 + 8;
    v20 = v6;
    v19 = v7;
    *(_QWORD *)(a1 + 220) = *(_QWORD *)(a2 + 220);
    if ( (_DWORD)v8 )
    {
      v10 = 0;
      v21 = 8LL * (unsigned int)(v8 - 1);
      v11 = v7;
      while ( 1 )
      {
        v16 = *(size_t **)(v11 + v10);
        v17 = (size_t **)(v6 + v10);
        if ( v16 == (size_t *)-8LL || !v16 )
        {
          *v17 = v16;
        }
        else
        {
          v22 = *(_QWORD *)(v11 + v10);
          n = *v16;
          v12 = sub_C7D670(*v16 + 17, 8);
          v13 = n;
          v14 = v22;
          v15 = v12;
          if ( n )
          {
            v18 = v12;
            memcpy((void *)(v12 + 16), (const void *)(v22 + 16), n);
            v13 = n;
            v14 = v22;
            v15 = v18;
          }
          *(_BYTE *)(v15 + v13 + 16) = 0;
          *(_QWORD *)v15 = v13;
          *(_DWORD *)(v15 + 8) = *(_DWORD *)(v14 + 8);
          *v17 = (size_t *)v15;
          *(_DWORD *)(v20 + v9) = *(_DWORD *)(v19 + v9);
        }
        v9 += 4;
        if ( v21 == v10 )
          break;
        v11 = *(_QWORD *)(a2 + 208);
        v6 = *(_QWORD *)(a1 + 208);
        v10 += 8;
      }
    }
  }
  result = *(unsigned __int8 *)(a2 + 232);
  *(_BYTE *)(a1 + 232) = result;
  return result;
}
