// Function: sub_11E9290
// Address: 0x11e9290
//
__int64 __fastcall sub_11E9290(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  unsigned int v12; // r8d
  __int64 *v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  unsigned int v16; // r8d
  __int64 v17; // r14
  __int64 v18; // rdx
  __int64 v19; // rbx
  __int64 v20; // rax
  bool v21; // zf
  _QWORD *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // rbx
  __int64 v28; // r12
  __int64 v29; // rdx
  unsigned int v30; // esi
  __int64 v31; // r14
  __int64 v32; // rdx
  __int64 v33; // rbx
  __int64 v34; // rax
  _QWORD *v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rdi
  __int64 v39; // rcx
  __int64 v40; // rbx
  __int64 v41; // r12
  __int64 v42; // rdx
  unsigned int v43; // esi
  __int64 v44; // [rsp+0h] [rbp-70h]
  __int64 *v45; // [rsp+8h] [rbp-68h]
  _BYTE v46[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v47; // [rsp+30h] [rbp-40h]

  v5 = sub_B43CA0(a2);
  v6 = *(_QWORD *)(a2 - 32);
  v45 = (__int64 *)v5;
  if ( !v6 || *(_BYTE *)v6 || (v44 = *(_QWORD *)(v6 + 24), *(_QWORD *)(a2 + 80) != v44) )
    BUG();
  v7 = sub_11E8E10(a1, (unsigned __int8 *)a2, a3);
  if ( !v7 )
  {
    v9 = *(__int64 **)(a1 + 24);
    if ( sub_11C99B0(v45, v9, 0x100u) && !(unsigned __int8)sub_11DAC90((char *)a2, (__int64)v9, v10, v11, v12) )
    {
      v31 = sub_11C96C0((__int64)v45, *(__int64 **)(a1 + 24), 0x100u, v44, *(_QWORD *)(v6 + 120));
      v33 = v32;
      v34 = sub_B47F80((_BYTE *)a2);
      v21 = *(_QWORD *)(v34 - 32) == 0;
      *(_QWORD *)(v34 + 80) = v31;
      v7 = v34;
      if ( !v21 )
      {
        v35 = *(_QWORD **)(v34 - 16);
        v36 = *(_QWORD *)(v34 - 24);
        *v35 = v36;
        if ( v36 )
          *(_QWORD *)(v36 + 16) = *(_QWORD *)(v7 - 16);
      }
      *(_QWORD *)(v7 - 32) = v33;
      if ( v33 )
      {
        v37 = *(_QWORD *)(v33 + 16);
        *(_QWORD *)(v7 - 24) = v37;
        if ( v37 )
          *(_QWORD *)(v37 + 16) = v7 - 24;
        *(_QWORD *)(v7 - 16) = v33 + 16;
        *(_QWORD *)(v33 + 16) = v7 - 32;
      }
      v38 = *(_QWORD *)(a3 + 88);
      v39 = *(_QWORD *)(a3 + 56);
      v47 = 257;
      (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, _QWORD))(*(_QWORD *)v38 + 16LL))(
        v38,
        v7,
        v46,
        v39,
        *(_QWORD *)(a3 + 64));
      v40 = *(_QWORD *)a3;
      v41 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
      if ( *(_QWORD *)a3 != v41 )
      {
        do
        {
          v42 = *(_QWORD *)(v40 + 8);
          v43 = *(_DWORD *)v40;
          v40 += 16;
          sub_B99FD0(v7, v43, v42);
        }
        while ( v41 != v40 );
      }
    }
    else
    {
      v13 = *(__int64 **)(a1 + 24);
      if ( sub_11C99B0(v45, v13, 0x88u) && !(unsigned __int8)sub_11DA5F0((char *)a2, (__int64)v13, v14, v15, v16) )
      {
        v17 = sub_11C96C0((__int64)v45, *(__int64 **)(a1 + 24), 0x88u, v44, *(_QWORD *)(v6 + 120));
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
        v47 = 257;
        (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, _QWORD))(*(_QWORD *)v25 + 16LL))(
          v25,
          v7,
          v46,
          v26,
          *(_QWORD *)(a3 + 64));
        v27 = *(_QWORD *)a3;
        v28 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
        if ( *(_QWORD *)a3 != v28 )
        {
          do
          {
            v29 = *(_QWORD *)(v27 + 8);
            v30 = *(_DWORD *)v27;
            v27 += 16;
            sub_B99FD0(v7, v30, v29);
          }
          while ( v28 != v27 );
        }
      }
    }
  }
  return v7;
}
