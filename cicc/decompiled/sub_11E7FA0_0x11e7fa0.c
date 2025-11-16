// Function: sub_11E7FA0
// Address: 0x11e7fa0
//
unsigned __int64 __fastcall sub_11E7FA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r13
  unsigned __int64 v7; // r15
  __int64 *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  unsigned int v12; // r8d
  __int64 *v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  unsigned int v16; // r8d
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rbx
  __int64 v20; // rax
  bool v21; // zf
  _QWORD *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // rbx
  __int64 v29; // r12
  __int64 v30; // rdx
  unsigned int v31; // esi
  __int64 v32; // r13
  __int64 v33; // rdx
  __int64 v34; // rbx
  __int64 v35; // rax
  _QWORD *v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rcx
  __int64 v41; // rbx
  __int64 v42; // r12
  __int64 v43; // rdx
  unsigned int v44; // esi
  __int64 v45; // [rsp+10h] [rbp-70h]
  __int64 *v46; // [rsp+18h] [rbp-68h]
  int v47[8]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v48; // [rsp+40h] [rbp-40h]

  v5 = sub_B43CA0(a2);
  v6 = *(_QWORD *)(a2 - 32);
  v46 = (__int64 *)v5;
  if ( !v6 || *(_BYTE *)v6 || (v45 = *(_QWORD *)(v6 + 24), *(_QWORD *)(a2 + 80) != v45) )
    BUG();
  v7 = sub_11E74E0(a1, (unsigned __int8 *)a2, a3);
  if ( !v7 )
  {
    *(_QWORD *)v47 = 0x100000000LL;
    sub_11DA4B0(a2, v47, 2);
    v9 = *(__int64 **)(a1 + 24);
    if ( sub_11C99B0(v46, v9, 0x1BDu) && !(unsigned __int8)sub_11DAC90((char *)a2, (__int64)v9, v10, v11, v12) )
    {
      v32 = sub_11C96C0((__int64)v46, *(__int64 **)(a1 + 24), 0x1BDu, v45, *(_QWORD *)(v6 + 120));
      v34 = v33;
      v35 = sub_B47F80((_BYTE *)a2);
      v21 = *(_QWORD *)(v35 - 32) == 0;
      *(_QWORD *)(v35 + 80) = v32;
      v7 = v35;
      if ( !v21 )
      {
        v36 = *(_QWORD **)(v35 - 16);
        v37 = *(_QWORD *)(v35 - 24);
        *v36 = v37;
        if ( v37 )
          *(_QWORD *)(v37 + 16) = *(_QWORD *)(v7 - 16);
      }
      *(_QWORD *)(v7 - 32) = v34;
      if ( v34 )
      {
        v38 = *(_QWORD *)(v34 + 16);
        *(_QWORD *)(v7 - 24) = v38;
        if ( v38 )
          *(_QWORD *)(v38 + 16) = v7 - 24;
        *(_QWORD *)(v7 - 16) = v34 + 16;
        *(_QWORD *)(v34 + 16) = v7 - 32;
      }
      v39 = *(_QWORD *)(a3 + 88);
      v40 = *(_QWORD *)(a3 + 56);
      v48 = 257;
      (*(void (__fastcall **)(__int64, unsigned __int64, int *, __int64, _QWORD))(*(_QWORD *)v39 + 16LL))(
        v39,
        v7,
        v47,
        v40,
        *(_QWORD *)(a3 + 64));
      v41 = *(_QWORD *)a3;
      v42 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
      while ( v42 != v41 )
      {
        v43 = *(_QWORD *)(v41 + 8);
        v44 = *(_DWORD *)v41;
        v41 += 16;
        sub_B99FD0(v7, v44, v43);
      }
    }
    else
    {
      v13 = *(__int64 **)(a1 + 24);
      if ( sub_11C99B0(v46, v13, 0x8Au) && !(unsigned __int8)sub_11DA5F0((char *)a2, (__int64)v13, v14, v15, v16) )
      {
        v17 = sub_11C96C0((__int64)v46, *(__int64 **)(a1 + 24), 0x8Au, v45, *(_QWORD *)(v6 + 120));
        v19 = v18;
        v20 = sub_B47F80((_BYTE *)a2);
        v21 = *(_QWORD *)(v20 - 32) == 0;
        *(_QWORD *)(v20 + 80) = v17;
        v7 = v20;
        if ( !v21 )
        {
          v22 = *(_QWORD **)(v20 - 16);
          v23 = *(_QWORD *)(v20 - 24);
          *v22 = v23;
          if ( v23 )
            *(_QWORD *)(v23 + 16) = *(_QWORD *)(v7 - 16);
        }
        *(_QWORD *)(v7 - 32) = v19;
        if ( v19 )
        {
          v24 = *(_QWORD *)(v19 + 16);
          *(_QWORD *)(v7 - 24) = v24;
          if ( v24 )
            *(_QWORD *)(v24 + 16) = v7 - 24;
          *(_QWORD *)(v7 - 16) = v19 + 16;
          *(_QWORD *)(v19 + 16) = v7 - 32;
        }
        v25 = *(_QWORD *)(a3 + 88);
        v26 = *(_QWORD *)(a3 + 56);
        v27 = *(_QWORD *)(a3 + 64);
        v48 = 257;
        (*(void (__fastcall **)(__int64, unsigned __int64, int *, __int64, __int64))(*(_QWORD *)v25 + 16LL))(
          v25,
          v7,
          v47,
          v26,
          v27);
        v28 = *(_QWORD *)a3;
        v29 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
        while ( v29 != v28 )
        {
          v30 = *(_QWORD *)(v28 + 8);
          v31 = *(_DWORD *)v28;
          v28 += 16;
          sub_B99FD0(v7, v31, v30);
        }
      }
    }
  }
  return v7;
}
