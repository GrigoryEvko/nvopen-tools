// Function: sub_F80B30
// Address: 0xf80b30
//
__int64 __fastcall sub_F80B30(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v8; // rbx
  __int64 v9; // rcx
  __int64 v10; // r10
  unsigned __int8 *v11; // rdx
  int v12; // eax
  __int64 v13; // r9
  bool v14; // al
  __int64 v16; // rax
  __int16 v17; // ax
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // r9
  char v24; // al
  __int64 v25; // r9
  __int64 v26; // rdx
  int v27; // r12d
  __int64 v28; // r12
  __int64 i; // r13
  __int64 v30; // rdx
  unsigned int v31; // esi
  __int64 v32; // [rsp+0h] [rbp-F0h]
  __int64 v33; // [rsp+8h] [rbp-E8h]
  __int64 v34; // [rsp+10h] [rbp-E0h]
  __int64 v35; // [rsp+18h] [rbp-D8h]
  __int64 v36; // [rsp+18h] [rbp-D8h]
  __int64 v37; // [rsp+18h] [rbp-D8h]
  __int64 v38; // [rsp+18h] [rbp-D8h]
  __int64 v39; // [rsp+18h] [rbp-D8h]
  __int64 v40; // [rsp+18h] [rbp-D8h]
  __int64 v41; // [rsp+18h] [rbp-D8h]
  __int64 v42; // [rsp+18h] [rbp-D8h]
  _QWORD v43[4]; // [rsp+20h] [rbp-D0h] BYREF
  __int16 v44; // [rsp+40h] [rbp-B0h]
  char v45[32]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v46; // [rsp+70h] [rbp-80h]
  __int64 v47; // [rsp+80h] [rbp-70h] BYREF
  _QWORD v48[4]; // [rsp+88h] [rbp-68h] BYREF
  __int16 v49; // [rsp+A8h] [rbp-48h]
  _QWORD v50[8]; // [rsp+B0h] [rbp-40h] BYREF

  v8 = *(_QWORD *)(a2 + 16);
  v9 = *(_QWORD *)(a1 + 576);
  if ( v8 )
  {
    v10 = a5 - 24;
    while ( 1 )
    {
      v11 = *(unsigned __int8 **)(v8 + 24);
      if ( a3 == *((_QWORD *)v11 + 1) )
      {
        v12 = *v11;
        if ( (unsigned __int8)v12 > 0x1Cu && (unsigned int)(v12 - 67) <= 0xC && a4 == v12 - 29 )
        {
          if ( !a5 )
            BUG();
          if ( *((_QWORD *)v11 + 5) == *(_QWORD *)(a5 + 16) && (!v9 || v11 != (unsigned __int8 *)(v9 - 24)) )
          {
            v33 = a5;
            v13 = v10;
            v34 = v9;
            v35 = v10;
            if ( v11 == (unsigned __int8 *)v10 )
              return v13;
            v32 = *(_QWORD *)(v8 + 24);
            v14 = sub_B445A0((__int64)v11, v10);
            v10 = v35;
            v9 = v34;
            a5 = v33;
            if ( v14 )
              return v32;
          }
        }
      }
      v8 = *(_QWORD *)(v8 + 8);
      if ( !v8 )
        goto LABEL_16;
    }
  }
  else
  {
LABEL_16:
    v16 = *(_QWORD *)(a1 + 568);
    v48[0] = 0;
    v47 = a1 + 520;
    v48[1] = 0;
    v48[2] = v16;
    if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
    {
      v36 = a5;
      sub_BD73F0((__int64)v48);
      a5 = v36;
    }
    v17 = *(_WORD *)(a1 + 584);
    v37 = a5;
    v48[3] = *(_QWORD *)(a1 + 576);
    v49 = v17;
    sub_B33910(v50, (__int64 *)(a1 + 520));
    v19 = *(unsigned int *)(a1 + 792);
    v50[1] = a1;
    v20 = v37;
    if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 796) )
    {
      sub_C8D5F0(a1 + 784, (const void *)(a1 + 800), v19 + 1, 8u, v37, v18);
      v19 = *(unsigned int *)(a1 + 792);
      v20 = v37;
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 784) + 8 * v19) = &v47;
    v21 = a1 + 520;
    ++*(_DWORD *)(a1 + 792);
    if ( v20 )
      v20 -= 24;
    sub_D5F1F0(v21, v20);
    v44 = 261;
    v43[0] = sub_BD5D20(a2);
    v43[1] = v22;
    if ( a3 == *(_QWORD *)(a2 + 8) )
    {
      v23 = a2;
    }
    else
    {
      v23 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(a1 + 600) + 120LL))(
              *(_QWORD *)(a1 + 600),
              a4,
              a2,
              a3);
      if ( !v23 )
      {
        v46 = 257;
        v39 = sub_B51D30(a4, a2, a3, (__int64)v45, 0, 0);
        v24 = sub_920620(v39);
        v25 = v39;
        if ( v24 )
        {
          v26 = *(_QWORD *)(a1 + 616);
          v27 = *(_DWORD *)(a1 + 624);
          if ( v26 )
          {
            sub_B99FD0(v39, 3u, v26);
            v25 = v39;
          }
          v40 = v25;
          sub_B45150(v25, v27);
          v25 = v40;
        }
        v41 = v25;
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 608) + 16LL))(
          *(_QWORD *)(a1 + 608),
          v25,
          v43,
          *(_QWORD *)(a1 + 576),
          *(_QWORD *)(a1 + 584));
        v28 = *(_QWORD *)(a1 + 520);
        v23 = v41;
        for ( i = v28 + 16LL * *(unsigned int *)(a1 + 528); i != v28; v23 = v42 )
        {
          v30 = *(_QWORD *)(v28 + 8);
          v31 = *(_DWORD *)v28;
          v28 += 16;
          v42 = v23;
          sub_B99FD0(v23, v31, v30);
        }
      }
    }
    v38 = v23;
    sub_F80960((__int64)&v47);
    return v38;
  }
}
