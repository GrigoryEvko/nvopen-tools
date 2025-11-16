// Function: sub_279C520
// Address: 0x279c520
//
__int64 __fastcall sub_279C520(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned __int16 v4; // ax
  unsigned __int64 v6; // rcx
  unsigned int v7; // r13d
  __int64 v8; // r15
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  _QWORD *v12; // rdi
  int v13; // ecx
  __int64 v14; // rsi
  __int64 v15; // rcx
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // r8
  __int64 v19; // rsi
  __int64 *v20; // rbx
  __int64 v21; // r12
  __int64 v22; // rax
  unsigned __int64 *v23; // rbx
  unsigned __int64 *v24; // r12
  unsigned __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 v27; // rcx
  int v28; // edx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rax
  int v35; // eax
  __int64 v36; // [rsp+0h] [rbp-230h] BYREF
  __int64 v37; // [rsp+8h] [rbp-228h] BYREF
  _QWORD v38[2]; // [rsp+10h] [rbp-220h] BYREF
  _BYTE *v39[4]; // [rsp+20h] [rbp-210h] BYREF
  unsigned __int8 v40; // [rsp+40h] [rbp-1F0h]
  _QWORD v41[10]; // [rsp+50h] [rbp-1E0h] BYREF
  unsigned __int64 *v42; // [rsp+A0h] [rbp-190h]
  unsigned int v43; // [rsp+A8h] [rbp-188h]
  char v44; // [rsp+B0h] [rbp-180h] BYREF

  v3 = *(_QWORD *)(a1 + 16);
  if ( !v3 )
    return 0;
  v4 = *(_WORD *)(a2 + 2);
  if ( ((v4 >> 7) & 6) != 0 || (v4 & 1) != 0 )
    return 0;
  if ( !*(_QWORD *)(a2 + 16) )
  {
    sub_278A7A0(a1 + 136, (_BYTE *)a2);
    v34 = *(unsigned int *)(a1 + 656);
    if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 660) )
    {
      sub_C8D5F0(a1 + 648, (const void *)(a1 + 664), v34 + 1, 8u, v32, v33);
      v34 = *(unsigned int *)(a1 + 656);
    }
    v7 = 1;
    *(_QWORD *)(*(_QWORD *)(a1 + 648) + 8 * v34) = a2;
    ++*(_DWORD *)(a1 + 656);
    return v7;
  }
  v6 = sub_1037A30(v3, (unsigned __int8 *)a2, 1);
  if ( (v6 & 7) != 3 )
  {
    if ( (v6 & 7) - 1 <= 1 )
    {
      sub_278D170((__int64)v39, a1, a2, v6, *(_QWORD *)(a2 - 32));
      v7 = v40;
      if ( v40 )
      {
        v8 = sub_278B200(v39, a2, a2);
        sub_30EC4B0(*(_QWORD *)(a1 + 104), a2);
        sub_BD84D0(a2, v8);
        sub_278A7A0(a1 + 136, (_BYTE *)a2);
        v11 = *(unsigned int *)(a1 + 656);
        if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 660) )
        {
          sub_C8D5F0(a1 + 648, (const void *)(a1 + 664), v11 + 1, 8u, v9, v10);
          v11 = *(unsigned int *)(a1 + 656);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 648) + 8 * v11) = a2;
        v12 = *(_QWORD **)(a1 + 120);
        ++*(_DWORD *)(a1 + 656);
        if ( v12 )
        {
          v13 = *(_DWORD *)(*v12 + 56LL);
          v14 = *(_QWORD *)(*v12 + 40LL);
          if ( v13 )
          {
            v15 = (unsigned int)(v13 - 1);
            v16 = v15 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
            v17 = (__int64 *)(v14 + 16LL * v16);
            v18 = *v17;
            if ( a2 == *v17 )
            {
LABEL_13:
              v19 = v17[1];
              if ( v19 )
                sub_D6E4B0(v12, v19, 0, v15, v18, v10);
            }
            else
            {
              v35 = 1;
              while ( v18 != -4096 )
              {
                v10 = (unsigned int)(v35 + 1);
                v16 = v15 & (v35 + v16);
                v17 = (__int64 *)(v14 + 16LL * v16);
                v18 = *v17;
                if ( a2 == *v17 )
                  goto LABEL_13;
                v35 = v10;
              }
            }
          }
        }
        v20 = *(__int64 **)(a1 + 96);
        v36 = a2;
        v38[0] = &v36;
        v37 = v8;
        v38[1] = &v37;
        v21 = *v20;
        v22 = sub_B2BE50(*v20);
        if ( sub_B6EA50(v22)
          || (v30 = sub_B2BE50(v21),
              v31 = sub_B6F970(v30),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v31 + 48LL))(v31)) )
        {
          sub_2790350((__int64)v41, (__int64)v38);
          sub_1049740(v20, (__int64)v41);
          v23 = v42;
          v41[0] = &unk_49D9D40;
          v24 = &v42[10 * v43];
          if ( v42 != v24 )
          {
            do
            {
              v24 -= 10;
              v25 = v24[4];
              if ( (unsigned __int64 *)v25 != v24 + 6 )
                j_j___libc_free_0(v25);
              if ( (unsigned __int64 *)*v24 != v24 + 2 )
                j_j___libc_free_0(*v24);
            }
            while ( v23 != v24 );
            v24 = v42;
          }
          if ( v24 != (unsigned __int64 *)&v44 )
            _libc_free((unsigned __int64)v24);
        }
        v26 = *(_QWORD *)(a1 + 16);
        if ( v26 )
        {
          v27 = *(_QWORD *)(v8 + 8);
          v28 = *(unsigned __int8 *)(v27 + 8);
          if ( (unsigned int)(v28 - 17) <= 1 )
            LOBYTE(v28) = *(_BYTE *)(**(_QWORD **)(v27 + 16) + 8LL);
          if ( (_BYTE)v28 == 14 )
            sub_102B9D0(v26, v8);
        }
        return v7;
      }
    }
    return 0;
  }
  if ( v6 >> 61 != 1 )
    return 0;
  return sub_279C4C0(a1, a2);
}
