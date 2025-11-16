// Function: sub_D36600
// Address: 0xd36600
//
void __fastcall sub_D36600(__int64 a1)
{
  __int64 v1; // r15
  int *v2; // r12
  __int64 v3; // rax
  int *v4; // r13
  signed __int64 v5; // rax
  int *v6; // r14
  int v7; // eax
  int v8; // r13d
  __int64 v9; // rax
  const char *v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rdx
  _QWORD *v13; // rax
  size_t v14; // rax
  int v15; // r12d
  __int8 *v16; // rax
  __int64 v17; // r12
  __int64 v18; // rbx
  __int64 v19; // rsi
  __int64 v20; // rsi
  char *v21; // rbx
  char v22; // al
  const char *v23; // [rsp+8h] [rbp-128h]
  unsigned int v24; // [rsp+14h] [rbp-11Ch]
  unsigned int v25; // [rsp+18h] [rbp-118h]
  __int64 v26; // [rsp+20h] [rbp-110h] BYREF
  __int64 v27; // [rsp+28h] [rbp-108h] BYREF
  __int64 v28; // [rsp+30h] [rbp-100h]
  __int64 v29; // [rsp+38h] [rbp-F8h]
  __int8 *v30; // [rsp+40h] [rbp-F0h] BYREF
  size_t v31; // [rsp+48h] [rbp-E8h]
  _QWORD v32[2]; // [rsp+50h] [rbp-E0h] BYREF
  _BYTE *v33[2]; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v34; // [rsp+70h] [rbp-C0h] BYREF
  __int64 *v35; // [rsp+80h] [rbp-B0h]
  __int64 v36; // [rsp+88h] [rbp-A8h]
  __int64 v37; // [rsp+90h] [rbp-A0h] BYREF
  __m128i v38; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v39[2]; // [rsp+B0h] [rbp-80h] BYREF
  _QWORD v40[2]; // [rsp+C0h] [rbp-70h] BYREF
  _QWORD *v41; // [rsp+D0h] [rbp-60h] BYREF
  _QWORD v42[2]; // [rsp+E0h] [rbp-50h] BYREF
  __m128i v43; // [rsp+F0h] [rbp-40h]

  v1 = *(_QWORD *)(a1 + 16);
  if ( *(_BYTE *)(v1 + 232) )
  {
    v2 = *(int **)(v1 + 240);
    v3 = 3LL * *(unsigned int *)(v1 + 248);
    v4 = &v2[v3];
    v5 = 0xAAAAAAAAAAAAAAABLL * ((v3 * 4) >> 2);
    if ( v5 >> 2 )
    {
      v6 = &v2[12 * (v5 >> 2)];
      while ( !(unsigned int)sub_D354B0(v2[2]) )
      {
        if ( (unsigned int)sub_D354B0(v2[5]) )
        {
          v4 = v2 + 3;
          goto LABEL_10;
        }
        if ( (unsigned int)sub_D354B0(v2[8]) )
        {
          v4 = v2 + 6;
          goto LABEL_10;
        }
        if ( (unsigned int)sub_D354B0(v2[11]) )
        {
          v4 = v2 + 9;
          goto LABEL_10;
        }
        v2 += 12;
        if ( v6 == v2 )
        {
          v5 = 0xAAAAAAAAAAAAAAABLL * (v4 - v2);
          goto LABEL_58;
        }
      }
      goto LABEL_9;
    }
LABEL_58:
    if ( v5 != 2 )
    {
      if ( v5 != 3 )
      {
        if ( v5 != 1 )
          goto LABEL_10;
        goto LABEL_61;
      }
      if ( (unsigned int)sub_D354B0(v2[2]) )
      {
LABEL_9:
        v4 = v2;
        goto LABEL_10;
      }
      v2 += 3;
    }
    if ( !(unsigned int)sub_D354B0(v2[2]) )
    {
      v2 += 3;
LABEL_61:
      if ( (unsigned int)sub_D354B0(v2[2]) )
        v4 = v2;
LABEL_10:
      if ( v4 == (int *)(*(_QWORD *)(v1 + 240) + 12LL * *(unsigned int *)(v1 + 248)) )
        return;
      v24 = *v4;
      v7 = v4[1];
      v8 = v4[2];
      v25 = v7;
      v9 = sub_D4A110(*(_QWORD *)(a1 + 24), "llvm.loop.distribute.enable", 27);
      v10 = "unsafe dependent memory operations in loop. Use #pragma clang loop distribute(enable) to allow loop distribu"
            "tion to attempt to isolate the offending operations into a separate loop";
      v29 = v11;
      v28 = v9;
      if ( (_BYTE)v11 )
      {
        v12 = *(_QWORD *)(*(_QWORD *)v28 + 136LL);
        v13 = *(_QWORD **)(v12 + 24);
        if ( *(_DWORD *)(v12 + 32) > 0x40u )
          v13 = (_QWORD *)*v13;
        v10 = "unsafe dependent memory operations in loop. Use #pragma clang loop distribute(enable) to allow loop distri"
              "bution to attempt to isolate the offending operations into a separate loop";
        if ( v13 )
          v10 = "unsafe dependent memory operations in loop.";
      }
      v23 = v10;
      v30 = (__int8 *)v32;
      v14 = strlen(v10);
      v15 = v14;
      v39[0] = v14;
      v16 = (__int8 *)sub_22409D0(&v30, v39, 0);
      v30 = v16;
      v32[0] = v39[0];
      *(_QWORD *)v16 = *(_QWORD *)v23;
      *(_QWORD *)&v16[v15 - 8] = *(_QWORD *)&v23[v15 - 8];
      qmemcpy(
        (void *)((unsigned __int64)(v16 + 8) & 0xFFFFFFFFFFFFFFF8LL),
        (const void *)(v23 - &v16[-((unsigned __int64)(v16 + 8) & 0xFFFFFFFFFFFFFFF8LL)]),
        8LL * ((v15 + (unsigned int)v16 - (((_DWORD)v16 + 8) & 0xFFFFFFF8)) >> 3));
      v31 = v39[0];
      v30[v39[0]] = 0;
      v17 = sub_D364E0(a1, (__int64)"UnsafeDep", 9, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 56LL) + 8LL * v25));
      sub_B18290(v17, v30, v31);
      switch ( v8 )
      {
        case 0:
        case 3:
        case 6:
          BUG();
        case 1:
          sub_B18290(v17, "\nUnknown data dependence.", 0x19u);
          break;
        case 2:
          sub_B18290(v17, "\nUnsafe indirect dependence.", 0x1Cu);
          break;
        case 4:
          sub_B18290(v17, "\nForward loop carried data dependence that prevents store-to-load forwarding.", 0x4Du);
          break;
        case 5:
          sub_B18290(v17, "\nBackward loop carried data dependence.", 0x27u);
          break;
        case 7:
          sub_B18290(v17, "\nBackward loop carried data dependence that prevents store-to-load forwarding.", 0x4Eu);
          break;
        default:
          break;
      }
      v21 = *(char **)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 56LL) + 8LL * v24);
      if ( !v21 )
        goto LABEL_39;
      v20 = *((_QWORD *)v21 + 6);
      v26 = v20;
      if ( v20 )
      {
        sub_B96E90((__int64)&v26, v20, 1);
        v22 = *v21;
        v20 = v26;
        if ( (unsigned __int8)*v21 <= 0x1Cu )
          goto LABEL_24;
      }
      else
      {
        v22 = *v21;
        if ( (unsigned __int8)*v21 <= 0x1Cu )
        {
LABEL_39:
          if ( v30 != (__int8 *)v32 )
            j_j___libc_free_0(v30, v32[0] + 1LL);
          return;
        }
      }
      if ( v22 == 61 || v22 == 62 )
      {
        v18 = *((_QWORD *)v21 - 4);
        if ( !v18 )
          goto LABEL_24;
      }
      else if ( v22 != 63 || (v18 = *(_QWORD *)&v21[-32 * (*((_DWORD *)v21 + 1) & 0x7FFFFFF)]) == 0 )
      {
LABEL_24:
        if ( v20 )
        {
          sub_B18290(v17, " Memory location is the same as accessed at ", 0x2Cu);
          v27 = v26;
          if ( v26 )
            sub_B96E90((__int64)&v27, v26, 1);
          sub_B16E20((__int64)v33, "Location", 8, &v27);
          v39[0] = (__int64)v40;
          sub_D32550(v39, v33[0], (__int64)&v33[0][(unsigned __int64)v33[1]]);
          v41 = v42;
          sub_D32550((__int64 *)&v41, v35, (__int64)v35 + v36);
          v43 = _mm_loadu_si128(&v38);
          sub_B180C0(v17, (unsigned __int64)v39);
          if ( v41 != v42 )
            j_j___libc_free_0(v41, v42[0] + 1LL);
          if ( (_QWORD *)v39[0] != v40 )
            j_j___libc_free_0(v39[0], v40[0] + 1LL);
          if ( v35 != &v37 )
            j_j___libc_free_0(v35, v37 + 1);
          if ( (__int64 *)v33[0] != &v34 )
            j_j___libc_free_0(v33[0], v34 + 1);
          if ( v27 )
            sub_B91220((__int64)&v27, v27);
          if ( v26 )
            sub_B91220((__int64)&v26, v26);
        }
        goto LABEL_39;
      }
      if ( *(_BYTE *)v18 > 0x1Cu && &v26 != (__int64 *)(v18 + 48) )
      {
        if ( v20 )
          sub_B91220((__int64)&v26, v20);
        v19 = *(_QWORD *)(v18 + 48);
        v26 = v19;
        if ( !v19 )
          goto LABEL_39;
        sub_B96E90((__int64)&v26, v19, 1);
        v20 = v26;
      }
      goto LABEL_24;
    }
    goto LABEL_9;
  }
}
