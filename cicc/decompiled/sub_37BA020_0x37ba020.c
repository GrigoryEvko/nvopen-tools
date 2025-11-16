// Function: sub_37BA020
// Address: 0x37ba020
//
void __fastcall sub_37BA020(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r8
  __int64 v7; // rdx
  __int64 *v8; // r12
  int v9; // edx
  int v10; // eax
  __int64 v11; // r8
  __int32 v12; // eax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r14
  _QWORD *v18; // rax
  bool v19; // al
  int v20; // [rsp-84h] [rbp-84h]
  _QWORD *v21; // [rsp-80h] [rbp-80h]
  __int64 v22; // [rsp-80h] [rbp-80h]
  __int64 v23; // [rsp-78h] [rbp-78h] BYREF
  __int16 v24; // [rsp-70h] [rbp-70h]
  __m128i v25; // [rsp-68h] [rbp-68h] BYREF
  __int64 v26; // [rsp-58h] [rbp-58h]
  __int64 v27; // [rsp-50h] [rbp-50h]
  __int64 v28; // [rsp-48h] [rbp-48h]

  if ( *(_BYTE *)(a1 + 40) )
  {
    v6 = *(_QWORD *)a3;
    if ( !*(_BYTE *)(a3 + 9)
      || (v25.m128i_i64[0] = sub_B0D520(*(_QWORD *)a3), v6 = v25.m128i_i64[0], v25.m128i_i64[1] = v7, (_BYTE)v7) )
    {
      v8 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 32LL) + 48LL * a2);
      if ( *(_WORD *)(*v8 + 20) )
      {
        if ( !v8[4] )
        {
          if ( !(unsigned int)((__int64)(*(_QWORD *)(v6 + 24) - *(_QWORD *)(v6 + 16)) >> 3)
            || (v22 = v6, v19 = sub_AF4770(v6), v6 = v22, v19) )
          {
            if ( (*(_DWORD *)a4 & 0xFFFFF) == 0 )
            {
              v21 = (_QWORD *)v6;
              if ( (*(_QWORD *)a4 & 0xFFFFF00000LL) == 0
                && *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 88LL) + 4LL * (*(_DWORD *)(a4 + 4) >> 8)) < *(_DWORD *)(*(_QWORD *)(a1 + 16) + 284LL) )
              {
                v20 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 104LL);
                v9 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 3616) + 680LL))(
                       *(_QWORD *)(a1 + 3616),
                       *(_QWORD *)(a1 + 24));
                v10 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 88LL) + 4LL * (*(_DWORD *)(a4 + 4) >> 8));
                if ( v20 != v10 && v10 != v9 )
                {
                  v11 = sub_B0DAC0(v21, 8, 0);
                  v12 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 88LL) + 4LL * (*(_DWORD *)(a4 + 4) >> 8));
                  v25.m128i_i64[0] = 0;
                  v25.m128i_i32[2] = v12;
                  LOBYTE(v12) = *(_BYTE *)(a3 + 8);
                  v26 = 0;
                  v27 = 0;
                  v28 = 0;
                  v23 = v11;
                  v24 = (unsigned __int8)v12;
                  sub_37B9D40((_QWORD *)a1, &v25, v8, &v23);
                  v15 = *(unsigned int *)(a1 + 3480);
                  v17 = v16;
                  if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 3484) )
                  {
                    sub_C8D5F0(a1 + 3472, (const void *)(a1 + 3488), v15 + 1, 0x10u, v13, v14);
                    v15 = *(unsigned int *)(a1 + 3480);
                  }
                  v18 = (_QWORD *)(*(_QWORD *)(a1 + 3472) + 16 * v15);
                  *v18 = a2;
                  v18[1] = v17;
                  ++*(_DWORD *)(a1 + 3480);
                }
              }
            }
          }
        }
      }
    }
  }
}
