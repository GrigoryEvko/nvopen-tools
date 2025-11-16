// Function: sub_38C4FC0
// Address: 0x38c4fc0
//
__int64 __fastcall sub_38C4FC0(__int64 a1, unsigned int *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // r14
  unsigned int *v7; // r13
  __int64 v8; // r15
  _QWORD *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // r15d
  __int64 v13; // r9
  __int64 v14; // r8
  int v15; // esi
  unsigned int v16; // r10d
  signed int v17; // eax
  int v18; // eax
  void (__fastcall *v19)(_QWORD *, __int64, __int64, __int64, _QWORD, __int64); // r8
  __int64 v20; // r15
  unsigned int v21; // r15d
  unsigned int v22; // r15d
  __int64 v23; // r8
  unsigned int v24; // r15d
  unsigned int v25; // r15d
  unsigned int v26; // r15d
  __int64 v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rsi
  unsigned int v30; // eax
  unsigned int v31; // eax
  int v32; // eax
  unsigned int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  unsigned int v37; // eax
  __int64 v38; // [rsp+8h] [rbp-48h]
  unsigned int v39; // [rsp+14h] [rbp-3Ch]
  unsigned int v40; // [rsp+14h] [rbp-3Ch]
  unsigned int v41; // [rsp+14h] [rbp-3Ch]
  unsigned int v42; // [rsp+14h] [rbp-3Ch]
  unsigned int v43; // [rsp+14h] [rbp-3Ch]
  unsigned int v44; // [rsp+14h] [rbp-3Ch]
  unsigned int *v45; // [rsp+18h] [rbp-38h]

  result = (__int64)&a2[12 * a3];
  v45 = (unsigned int *)result;
  if ( (unsigned int *)result != a2 )
  {
    v6 = a4;
    v7 = a2;
    while ( 1 )
    {
      v8 = *((_QWORD *)v7 + 1);
      if ( v8 )
      {
        if ( (*(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        {
          result = *(_BYTE *)(v8 + 9) & 0xC;
          if ( (_BYTE)result != 8 )
            goto LABEL_5;
          *(_BYTE *)(v8 + 8) |= 4u;
          v28 = sub_38CE440(*(_QWORD *)(v8 + 24));
          result = v28 | *(_QWORD *)v8 & 7LL;
          *(_QWORD *)v8 = result;
          if ( !v28 )
            goto LABEL_5;
        }
        if ( v6 && v8 != v6 )
        {
          v29 = v6;
          v6 = v8;
          sub_38D5C00(*(_QWORD *)(a1 + 16), v29, v8);
        }
      }
      v9 = *(_QWORD **)(a1 + 16);
      v10 = v9[1];
      v11 = *(_QWORD *)(v10 + 16);
      v12 = *(_DWORD *)(v11 + 12);
      if ( !*(_BYTE *)(v11 + 17) )
        v12 = -v12;
      v13 = *(_QWORD *)(v10 + 24);
      v14 = *v7;
      switch ( *v7 )
      {
        case 0u:
          v22 = v7[4];
          (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v9 + 424LL))(
            v9,
            8,
            1,
            a4,
            v14,
            v13);
          result = sub_38DCDD0(*(_QWORD *)(a1 + 16), v22);
          goto LABEL_5;
        case 1u:
          result = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v9 + 424LL))(
                     v9,
                     10,
                     1,
                     a4,
                     v14,
                     v13);
          goto LABEL_5;
        case 2u:
          result = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v9 + 424LL))(
                     v9,
                     11,
                     1,
                     a4,
                     v14,
                     v13);
          goto LABEL_5;
        case 3u:
        case 7u:
          v16 = v7[4];
          if ( *(_BYTE *)(a1 + 8) )
          {
            v17 = v7[5];
            if ( (_DWORD)v14 != 7 )
              goto LABEL_14;
LABEL_44:
            v17 -= *(_DWORD *)a1;
            goto LABEL_14;
          }
          v42 = *v7;
          v30 = sub_38D7200(v13, v16, v11, a4, v14);
          v9 = *(_QWORD **)(a1 + 16);
          v16 = v30;
          v17 = v7[5];
          if ( v42 == 7 )
            goto LABEL_44;
LABEL_14:
          v18 = v17 / v12;
          v19 = *(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, _QWORD, __int64))(*v9 + 424LL);
          v20 = v18;
          if ( v18 < 0 )
          {
            v41 = v16;
            v19(v9, 17, 1, a4, v19, v13);
            sub_38DCDD0(*(_QWORD *)(a1 + 16), v41);
            result = sub_38DCF20(*(_QWORD *)(a1 + 16), (int)v20);
LABEL_5:
            v7 += 12;
            if ( v45 == v7 )
              return result;
          }
          else
          {
            if ( v16 > 0x3F )
            {
              v44 = v16;
              v19(v9, 5, 1, a4, v19, v13);
              sub_38DCDD0(*(_QWORD *)(a1 + 16), v44);
              result = sub_38DCDD0(*(_QWORD *)(a1 + 16), v20);
              goto LABEL_5;
            }
            v7 += 12;
            v19(v9, v16 + 128, 1, a4, v19, v13);
            result = sub_38DCDD0(*(_QWORD *)(a1 + 16), v20);
            if ( v45 == v7 )
              return result;
          }
          break;
        case 4u:
          v21 = v7[4];
          if ( !*(_BYTE *)(a1 + 8) )
          {
            v33 = sub_38D7200(v13, v21, v11, a4, v14);
            v9 = *(_QWORD **)(a1 + 16);
            v21 = v33;
          }
          (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v9 + 424LL))(
            v9,
            13,
            1,
            a4,
            v14,
            v13);
          result = sub_38DCDD0(*(_QWORD *)(a1 + 16), v21);
          goto LABEL_5;
        case 5u:
        case 8u:
          v39 = *v7;
          (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v9 + 424LL))(
            v9,
            14,
            1,
            a4,
            v14,
            v13);
          if ( v39 != 8 )
            goto LABEL_34;
          v15 = v7[5] + *(_DWORD *)a1;
          goto LABEL_35;
        case 6u:
          v26 = v7[4];
          if ( !*(_BYTE *)(a1 + 8) )
          {
            v31 = sub_38D7200(v13, v26, v11, a4, v14);
            v9 = *(_QWORD **)(a1 + 16);
            v26 = v31;
          }
          (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v9 + 424LL))(
            v9,
            12,
            1,
            a4,
            v14,
            v13);
          sub_38DCDD0(*(_QWORD *)(a1 + 16), v26);
LABEL_34:
          v15 = -v7[5];
LABEL_35:
          *(_DWORD *)a1 = v15;
          result = sub_38DCDD0(*(_QWORD *)(a1 + 16), v15);
          goto LABEL_5;
        case 9u:
          result = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, _QWORD, __int64, __int64, __int64))(*v9 + 400LL))(
                     v9,
                     *((_QWORD *)v7 + 3),
                     *((_QWORD *)v7 + 4) - *((_QWORD *)v7 + 3),
                     a4,
                     v14,
                     v13);
          goto LABEL_5;
        case 0xAu:
          v27 = v7[4];
          if ( !*(_BYTE *)(a1 + 8) )
          {
            v32 = sub_38D7200(v13, v27, v11, a4, v14);
            v9 = *(_QWORD **)(a1 + 16);
            LODWORD(v27) = v32;
          }
          result = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, __int64, __int64, __int64, __int64))(*v9 + 424LL))(
                     v9,
                     (unsigned int)v27 | 0xC0,
                     1,
                     a4,
                     v14,
                     v13);
          goto LABEL_5;
        case 0xBu:
          v25 = v7[4];
          (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v9 + 424LL))(
            v9,
            7,
            1,
            a4,
            v14,
            v13);
          result = sub_38DCDD0(*(_QWORD *)(a1 + 16), v25);
          goto LABEL_5;
        case 0xCu:
          v23 = v7[4];
          v24 = v7[5];
          if ( !*(_BYTE *)(a1 + 8) )
          {
            v38 = *(_QWORD *)(v10 + 24);
            v43 = sub_38D7200(v13, (unsigned int)v23, v11, a4, v23);
            v37 = sub_38D7200(v38, v24, v34, v35, v36);
            v9 = *(_QWORD **)(a1 + 16);
            v23 = v43;
            v24 = v37;
          }
          v40 = v23;
          (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v9 + 424LL))(
            v9,
            9,
            1,
            a4,
            v23,
            v13);
          sub_38DCDD0(*(_QWORD *)(a1 + 16), v40);
          result = sub_38DCDD0(*(_QWORD *)(a1 + 16), v24);
          goto LABEL_5;
        case 0xDu:
          result = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v9 + 424LL))(
                     v9,
                     45,
                     1,
                     a4,
                     v14,
                     v13);
          goto LABEL_5;
        case 0xEu:
          (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v9 + 424LL))(
            v9,
            46,
            1,
            a4,
            v14,
            v13);
          result = sub_38DCDD0(*(_QWORD *)(a1 + 16), (int)v7[5]);
          goto LABEL_5;
      }
    }
  }
  return result;
}
