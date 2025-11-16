// Function: sub_38170E0
// Address: 0x38170e0
//
__int64 *__fastcall sub_38170E0(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // rax
  __int16 v8; // dx
  int v9; // eax
  __int64 v10; // rcx
  __int64 v11; // r8
  unsigned int v12; // edx
  __int64 *result; // rax
  __int64 v14; // rdx
  __int64 v15; // r9
  unsigned __int8 *v16; // r10
  __int64 v17; // r11
  __int64 v18; // rax
  __int64 v19; // rsi
  _QWORD *v20; // r15
  __int64 v21; // r14
  unsigned int v22; // ebx
  __int64 v23; // r8
  unsigned int v24; // esi
  unsigned __int16 v25; // r15
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // rdx
  __int128 v30; // [rsp-30h] [rbp-B0h]
  unsigned __int8 *v31; // [rsp+0h] [rbp-80h]
  __int64 v32; // [rsp+8h] [rbp-78h]
  __int64 v33; // [rsp+18h] [rbp-68h]
  __int64 *v34; // [rsp+18h] [rbp-68h]
  unsigned __int8 *v35; // [rsp+20h] [rbp-60h]
  __int64 v36; // [rsp+30h] [rbp-50h] BYREF
  __int64 v37; // [rsp+38h] [rbp-48h]
  __int64 v38; // [rsp+40h] [rbp-40h] BYREF
  int v39; // [rsp+48h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)v4;
  v6 = *(_QWORD *)(v4 + 8);
  v7 = *(_QWORD *)(*(_QWORD *)(v4 + 40) + 48LL) + 16LL * *(unsigned int *)(v4 + 48);
  v8 = *(_WORD *)v7;
  v37 = *(_QWORD *)(v7 + 8);
  v9 = *(_DWORD *)(a2 + 24);
  LOWORD(v36) = v8;
  if ( v9 != 206 )
  {
LABEL_2:
    if ( v9 != 205 )
    {
      v10 = v36;
      v11 = v37;
LABEL_4:
      v35 = sub_375B580(a1, v5, a3, v6, v10, v11);
      return sub_33EC3B0(
               *(_QWORD **)(a1 + 8),
               (__int64 *)a2,
               (__int64)v35,
               v6 & 0xFFFFFFFF00000000LL | v12,
               *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
               *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
               *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
    }
    v25 = v36;
    if ( (_WORD)v36 )
    {
      if ( (unsigned __int16)(v36 - 17) <= 0xD3u )
      {
        v11 = 0;
        v25 = word_4456580[(unsigned __int16)v36 - 1];
        goto LABEL_15;
      }
    }
    else if ( sub_30070B0((__int64)&v36) )
    {
      v25 = sub_3009970((__int64)&v36, a2, v26, v27, v28);
      v11 = v29;
      goto LABEL_15;
    }
    v11 = v37;
LABEL_15:
    v10 = v25;
    goto LABEL_4;
  }
  v16 = sub_3791F80((__int64 *)a1, a2);
  v17 = v14;
  if ( !v16 )
  {
    v9 = *(_DWORD *)(a2 + 24);
    goto LABEL_2;
  }
  v18 = *(_QWORD *)(a2 + 48);
  v19 = *(_QWORD *)(a2 + 80);
  v20 = *(_QWORD **)(a1 + 8);
  v21 = *(_QWORD *)(a2 + 40);
  v22 = **(unsigned __int16 **)(a2 + 48);
  v23 = *(_QWORD *)(v18 + 8);
  v38 = v19;
  if ( v19 )
  {
    v32 = v14;
    v31 = v16;
    v33 = v23;
    sub_B96E90((__int64)&v38, v19, 1);
    v16 = v31;
    v17 = v32;
    v23 = v33;
  }
  v24 = *(_DWORD *)(a2 + 24);
  v39 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v30 + 1) = v17;
  *(_QWORD *)&v30 = v16;
  result = (__int64 *)sub_340F900(
                        v20,
                        v24,
                        (__int64)&v38,
                        v22,
                        v23,
                        v15,
                        v30,
                        *(_OWORD *)(v21 + 40),
                        *(_OWORD *)(v21 + 80));
  if ( v38 )
  {
    v34 = result;
    sub_B91220((__int64)&v38, v38);
    return v34;
  }
  return result;
}
