// Function: sub_2123540
// Address: 0x2123540
//
__int64 *__fastcall sub_2123540(__int64 *a1, __int64 a2, unsigned int a3, __m128 a4, double a5, __m128i a6)
{
  unsigned __int8 *v7; // rax
  __int64 v8; // r14
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // rdx
  unsigned int v11; // r14d
  unsigned __int64 v12; // rax
  __int64 v13; // rsi
  __int64 *v14; // r11
  unsigned __int64 *v15; // r15
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int8 v18; // r9
  const void **v19; // r8
  int v20; // eax
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // rdx
  unsigned int v23; // ebx
  unsigned __int64 v24; // r9
  __int16 *v25; // r10
  __int64 v26; // rax
  char v27; // si
  __int64 v28; // rax
  bool v29; // al
  unsigned int v30; // esi
  __int64 *v31; // rbx
  const void **v33; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v34; // [rsp+0h] [rbp-A0h]
  __int16 *v35; // [rsp+8h] [rbp-98h]
  unsigned __int8 v36; // [rsp+10h] [rbp-90h]
  unsigned __int64 v37; // [rsp+18h] [rbp-88h]
  __int64 *v38; // [rsp+20h] [rbp-80h]
  const void **v39; // [rsp+20h] [rbp-80h]
  unsigned __int64 v40; // [rsp+28h] [rbp-78h]
  __int64 *v41; // [rsp+28h] [rbp-78h]
  __int64 v42; // [rsp+30h] [rbp-70h]
  __int64 v43; // [rsp+30h] [rbp-70h]
  __int64 v44; // [rsp+38h] [rbp-68h]
  __int64 v45; // [rsp+40h] [rbp-60h] BYREF
  int v46; // [rsp+48h] [rbp-58h]
  _BYTE v47[8]; // [rsp+50h] [rbp-50h] BYREF
  __int64 v48; // [rsp+58h] [rbp-48h]
  __int64 v49; // [rsp+60h] [rbp-40h]

  v7 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * a3);
  v8 = *v7;
  v42 = *((_QWORD *)v7 + 1);
  sub_1F40D10((__int64)v47, *a1, *(_QWORD *)(a1[1] + 48), (unsigned __int8)v8, v42);
  if ( (_BYTE)v8 != (_BYTE)v48 )
    goto LABEL_2;
  if ( v42 == v49 )
  {
    if ( !(_BYTE)v8 )
      goto LABEL_2;
  }
  else if ( !(_BYTE)v8 )
  {
    goto LABEL_2;
  }
  if ( *(_QWORD *)(*a1 + 8 * v8 + 120) )
    return (__int64 *)a2;
LABEL_2:
  v9 = sub_2120330((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v11 = v10;
  v40 = v10;
  v12 = sub_2120330((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v13 = *(_QWORD *)(a2 + 72);
  v14 = (__int64 *)a1[1];
  v43 = v12;
  v15 = *(unsigned __int64 **)(a2 + 32);
  v16 = *(_QWORD *)(v9 + 40) + 16LL * v11;
  v44 = v17;
  v18 = *(_BYTE *)v16;
  v19 = *(const void ***)(v16 + 8);
  v45 = v13;
  if ( v13 )
  {
    v38 = v14;
    v33 = v19;
    v36 = v18;
    sub_1623A60((__int64)&v45, v13, 2);
    v19 = v33;
    v14 = v38;
    v18 = v36;
  }
  v20 = *(_DWORD *)(a2 + 64);
  v21 = v40;
  v22 = v9;
  v23 = v18;
  v46 = v20;
  v24 = *v15;
  v25 = (__int16 *)v15[1];
  v26 = *(_QWORD *)(*v15 + 40) + 16LL * *((unsigned int *)v15 + 2);
  v27 = *(_BYTE *)v26;
  v28 = *(_QWORD *)(v26 + 8);
  v47[0] = v27;
  v48 = v28;
  if ( v27 )
  {
    v30 = ((unsigned __int8)(v27 - 14) < 0x60u) + 134;
  }
  else
  {
    v34 = v24;
    v35 = v25;
    v37 = v40;
    v39 = v19;
    v41 = v14;
    v29 = sub_1F58D20((__int64)v47);
    v19 = v39;
    v24 = v34;
    v25 = v35;
    v14 = v41;
    v22 = v9;
    v21 = v37;
    v30 = 134 - (!v29 - 1);
  }
  v31 = sub_1D3A900(v14, v30, (__int64)&v45, v23, v19, 0, a4, a5, a6, v24, v25, __PAIR128__(v21, v22), v43, v44);
  if ( v45 )
    sub_161E7C0((__int64)&v45, v45);
  return v31;
}
