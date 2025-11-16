// Function: sub_83DAC0
// Address: 0x83dac0
//
__int64 __fastcall sub_83DAC0(
        __int64 a1,
        const __m128i *a2,
        int a3,
        int a4,
        __m128i *a5,
        __int64 *a6,
        __int64 *a7,
        _DWORD *a8)
{
  __int64 v8; // r15
  __m128i *v10; // r14
  __int64 result; // rax
  __int64 v12; // r13
  __int64 i; // rdi
  __int64 **v14; // rdx
  __int64 *v15; // r10
  __int64 v16; // rax
  __int64 v17; // r10
  __int64 v18; // r10
  __int64 *v19; // rax
  __int64 v20; // r11
  __int64 v21; // rax
  __int64 v22; // [rsp+0h] [rbp-50h]
  __int64 v23; // [rsp+0h] [rbp-50h]
  __int64 *v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+8h] [rbp-48h]
  __int64 **v26; // [rsp+8h] [rbp-48h]
  __int64 v27; // [rsp+8h] [rbp-48h]
  __int64 v28; // [rsp+8h] [rbp-48h]
  __int64 v29; // [rsp+8h] [rbp-48h]

  v8 = a1;
  *a6 = 0;
  *a7 = 0;
  if ( a8 )
    *a8 = 0;
  v10 = sub_73C570(a2, a3);
  result = *(unsigned __int8 *)(a1 + 80);
  if ( (_BYTE)result == 16 )
  {
    v8 = **(_QWORD **)(a1 + 88);
    result = *(unsigned __int8 *)(v8 + 80);
  }
  if ( (_BYTE)result == 24 )
  {
    v8 = *(_QWORD *)(v8 + 88);
    result = *(unsigned __int8 *)(v8 + 80);
  }
  v12 = *(_QWORD *)(v8 + 88);
  if ( (_BYTE)result != 20 )
  {
    v18 = *(_QWORD *)(v12 + 152);
    if ( (*(_BYTE *)(v8 + 104) & 1) != 0 )
    {
      v28 = *(_QWORD *)(v12 + 152);
      result = sub_8796F0(v8);
      v18 = v28;
    }
    else
    {
      result = (*(_BYTE *)(v12 + 208) & 4) != 0;
    }
    if ( (_DWORD)result )
      return result;
    goto LABEL_23;
  }
  v12 = *(_QWORD *)(v12 + 176);
  for ( i = *(_QWORD *)(v12 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v14 = *(__int64 ***)(i + 168);
  if ( a8 || (v26 = *(__int64 ***)(i + 168), result = sub_826C30(i, a4), v14 = v26, !(_DWORD)result) )
  {
    v15 = *v14;
    if ( *v14 )
    {
      result = *v15;
      if ( !*v15 || (*(_WORD *)(result + 32) & 0x104) != 0 )
      {
        v24 = *v14;
        v16 = sub_8D22E0(v15[1]);
        v17 = (__int64)v24;
        if ( (*(_BYTE *)(v16 + 140) & 0xFD) == 0xC || (result = sub_8D32E0(v16), v17 = (__int64)v24, (_DWORD)result) )
        {
          result = sub_83D790(v17, 0, v10, v8, a6);
          if ( (_DWORD)result )
          {
            result = sub_8B2240(a6, v8, 0, 0x20000, 0);
            v18 = result;
            if ( result )
            {
LABEL_23:
              while ( *(_BYTE *)(v18 + 140) == 12 )
                v18 = *(_QWORD *)(v18 + 160);
              *a7 = v18;
              v19 = *(__int64 **)(v18 + 168);
              v20 = *v19;
              if ( *(_BYTE *)(v12 + 174) != 1
                || (v23 = *v19,
                    v29 = v18,
                    result = sub_72F3C0(v18, (__int64)a2, 0, a4, 0),
                    v18 = v29,
                    v20 = v23,
                    (_DWORD)result) )
              {
                if ( !dword_4D04494 || *(_BYTE *)(v8 + 80) != 20 || !(_DWORD)qword_4F077B4 || !qword_4F077A0 )
                  goto LABEL_27;
                v21 = *(_QWORD *)(*(_QWORD *)(v8 + 88) + 176LL);
                if ( (*(_BYTE *)(v21 + 194) & 0x40) != 0 )
                {
                  do
                    v21 = *(_QWORD *)(v21 + 232);
                  while ( (*(_BYTE *)(v21 + 194) & 0x40) != 0 );
                  v8 = **(_QWORD **)(v21 + 248);
                }
                v22 = v18;
                v27 = v20;
                result = sub_8A00C0(v8, *a6, 0);
                v20 = v27;
                v18 = v22;
                if ( (_DWORD)result )
                {
LABEL_27:
                  v25 = v18;
                  result = sub_838020(0, (__int64)v10, *(__m128i **)(v20 + 8), v20, 0, 0, a5);
                  if ( a5->m128i_i32[2] != 7 && a8 )
                  {
                    result = sub_826C30(v25, a4);
                    if ( (_DWORD)result )
                    {
                      a5->m128i_i32[2] = 7;
                      *a8 = 1;
                      return (__int64)a8;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}
