// Function: sub_30460A0
// Address: 0x30460a0
//
__int64 __fastcall sub_30460A0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6, int a7)
{
  unsigned __int16 *v8; // rdx
  __int64 v9; // r15
  __int64 result; // rax
  char v14; // al
  int v15; // ecx
  __int64 v16; // rax
  int v17; // edx
  int v18; // esi
  int v19; // edx
  __int128 *v20; // rbx
  int v21; // eax
  __int64 v22; // rdi
  int v23; // eax
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // r9
  __int128 *v27; // r10
  __int64 v28; // rax
  __int128 v29; // [rsp-10h] [rbp-70h]
  int v30; // [rsp+0h] [rbp-60h]
  int v31; // [rsp+10h] [rbp-50h]
  int v32; // [rsp+10h] [rbp-50h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+20h] [rbp-40h] BYREF
  int v36; // [rsp+28h] [rbp-38h]

  v8 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
  v9 = *((_QWORD *)v8 + 1);
  if ( *(_DWORD *)(a2 + 24) != 98 )
    return 0;
  v31 = *v8;
  v14 = sub_3046050(*(_QWORD *)(*(_QWORD *)(a6 + 16) + 16LL), *(__int64 **)(*(_QWORD *)(a6 + 16) + 40LL), a7);
  v15 = v31;
  if ( !v14 )
    return 0;
  v16 = *(_QWORD *)(a2 + 56);
  if ( v16 )
  {
    v17 = 5;
    v18 = 0;
    while ( 1 )
    {
      v18 += *(_DWORD *)(*(_QWORD *)(v16 + 16) + 24LL) != 96;
      if ( !--v17 )
        return 0;
      v16 = *(_QWORD *)(v16 + 32);
      if ( !v16 )
      {
        if ( !v18 )
          break;
        v19 = *(_DWORD *)(a1 + 72);
        if ( v19 - *(_DWORD *)(a2 + 72) <= 499 )
          return 0;
        v20 = *(__int128 **)(a2 + 40);
        v21 = *(_DWORD *)(*(_QWORD *)v20 + 24LL);
        if ( v21 != 11 && v21 != 35 )
        {
          v22 = *((_QWORD *)v20 + 5);
          v23 = *(_DWORD *)(v22 + 24);
          if ( v23 != 11 && v23 != 35 )
          {
            v24 = *(_QWORD *)(*(_QWORD *)v20 + 56LL);
            if ( v24 )
            {
              while ( v19 >= *(_DWORD *)(*(_QWORD *)(v24 + 16) + 72LL) )
              {
                v24 = *(_QWORD *)(v24 + 32);
                if ( !v24 )
                  goto LABEL_24;
              }
            }
            else
            {
LABEL_24:
              v28 = *(_QWORD *)(v22 + 56);
              if ( !v28 )
                return 0;
              while ( v19 >= *(_DWORD *)(*(_QWORD *)(v28 + 16) + 72LL) )
              {
                v28 = *(_QWORD *)(v28 + 32);
                if ( !v28 )
                  return 0;
              }
            }
          }
        }
        goto LABEL_19;
      }
    }
  }
  v20 = *(__int128 **)(a2 + 40);
LABEL_19:
  v25 = *(_QWORD *)(a1 + 80);
  v26 = *(_QWORD *)(a6 + 16);
  v27 = (__int128 *)((char *)v20 + 40);
  v35 = v25;
  if ( v25 )
  {
    v30 = v31;
    v32 = v26;
    sub_B96E90((__int64)&v35, v25, 1);
    v15 = v30;
    v27 = (__int128 *)((char *)v20 + 40);
    LODWORD(v26) = v32;
  }
  *((_QWORD *)&v29 + 1) = a5;
  *(_QWORD *)&v29 = a4;
  v36 = *(_DWORD *)(a1 + 72);
  result = sub_340F900(v26, 150, (unsigned int)&v35, v15, v9, v26, *v20, *v27, v29);
  if ( v35 )
  {
    v33 = result;
    sub_B91220((__int64)&v35, v35);
    return v33;
  }
  return result;
}
