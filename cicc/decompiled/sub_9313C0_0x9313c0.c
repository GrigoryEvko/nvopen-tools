// Function: sub_9313C0
// Address: 0x9313c0
//
__int64 __fastcall sub_9313C0(__int64 a1, unsigned __int64 a2)
{
  __m128i *v3; // r14
  __int64 v4; // rdx
  int v5; // ecx
  __int64 v6; // rax
  unsigned __int8 v7; // al
  int v8; // edx
  int v9; // ecx
  int v10; // r8d
  __int64 v11; // rax
  int v12; // r9d
  __int64 v13; // r15
  unsigned int *v14; // rbx
  unsigned int *v15; // r14
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 *v21; // r14
  __int64 *i; // rbx
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // r15
  unsigned int v27; // eax
  bool v28; // al
  int v29; // [rsp+0h] [rbp-70h]
  int v30; // [rsp+4h] [rbp-6Ch]
  int v31; // [rsp+4h] [rbp-6Ch]
  int v32; // [rsp+8h] [rbp-68h]
  int v33; // [rsp+8h] [rbp-68h]
  __int64 *v34; // [rsp+10h] [rbp-60h] BYREF
  __int64 *v35; // [rsp+18h] [rbp-58h]
  __int64 v36; // [rsp+20h] [rbp-50h]
  __int16 v37; // [rsp+30h] [rbp-40h]

  v3 = *(__m128i **)(a2 + 48);
  if ( *(_QWORD *)(a1 + 208) )
  {
    if ( v3 )
    {
      if ( sub_91B770(v3->m128i_i64[0]) )
      {
        sub_947E80(a1, v3, *(_QWORD *)(a1 + 208), *(unsigned int *)(a1 + 216), 0);
      }
      else
      {
        v4 = *(_QWORD *)(a1 + 208);
        v5 = unk_4D0463C;
        if ( unk_4D0463C )
        {
          v28 = sub_90AA40(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 208));
          v4 = *(_QWORD *)(a1 + 208);
          v5 = v28;
        }
        v30 = v5;
        v32 = v4;
        if ( *(_DWORD *)(a1 + 216) )
        {
          _BitScanReverse64(&v26, *(unsigned int *)(a1 + 216));
          v27 = (unsigned int)sub_92F410(a1, (__int64)v3);
          v9 = v30;
          LODWORD(v3) = v27;
          v8 = v32;
          v10 = (unsigned __int8)(63 - (v26 ^ 0x3F));
        }
        else
        {
          v3 = sub_92F410(a1, (__int64)v3);
          v6 = sub_AA4E30(*(_QWORD *)(a1 + 96));
          v7 = sub_AE5020(v6, v3->m128i_i64[1]);
          v8 = v32;
          v9 = v30;
          v10 = v7;
        }
        v31 = v9;
        v29 = v10;
        v33 = v8;
        v37 = 257;
        v11 = sub_BD2C40(80, unk_3F10A10);
        v13 = v11;
        if ( v11 )
          sub_B4D3C0(v11, (_DWORD)v3, v33, v31, v29, v12, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, __int64 **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
          *(_QWORD *)(a1 + 136),
          v13,
          &v34,
          *(_QWORD *)(a1 + 104),
          *(_QWORD *)(a1 + 112));
        v14 = *(unsigned int **)(a1 + 48);
        v15 = &v14[4 * *(unsigned int *)(a1 + 56)];
        while ( v15 != v14 )
        {
          v16 = *((_QWORD *)v14 + 1);
          v17 = *v14;
          v14 += 4;
          sub_B99FD0(v13, v17, v16);
        }
      }
    }
  }
  else if ( v3 )
  {
    sub_921EA0((__int64)&v34, a1, *(__int64 **)(a2 + 48), 0, 0, 0);
  }
  if ( *(_BYTE *)(a1 + 240) )
  {
    sub_9310E0((__int64)&v34, a1, a2);
    v21 = v35;
    for ( i = v34; v21 != i; ++i )
    {
      v23 = *i;
      sub_91A3A0(*(_QWORD *)(a1 + 32) + 8LL, *(_QWORD *)(v23 + 120), v19, v20);
      sub_9465D0(a1, v23, 1);
    }
    v24 = *(_QWORD *)(a1 + 424);
    v25 = *(_QWORD *)(v24 - 24);
    if ( v25 != *(_QWORD *)(v24 - 16) )
      *(_QWORD *)(v24 - 16) = v25;
    if ( v34 )
      j_j___libc_free_0(v34, v36 - (_QWORD)v34);
  }
  return sub_92FD90(a1, *(_QWORD *)(a1 + 200));
}
