// Function: sub_291C070
// Address: 0x291c070
//
__int64 __fastcall sub_291C070(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v8; // r12
  char v10; // al
  _BYTE *v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rax
  char v14; // al
  __int64 v16; // rdi
  __int64 (__fastcall *v17)(__int64, unsigned __int64, __int64); // rax
  unsigned int *v18; // rbx
  __int64 v19; // r13
  __int64 v20; // rdx
  unsigned int v21; // esi
  unsigned int *v22; // rbx
  __int64 v23; // r13
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 *v26; // rdx
  _BYTE *v27; // rcx
  __int64 v28; // [rsp+8h] [rbp-A8h]
  _BYTE *v29; // [rsp+10h] [rbp-A0h]
  unsigned int v30; // [rsp+1Ch] [rbp-94h]
  _BYTE *v31[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v32; // [rsp+40h] [rbp-70h]
  _QWORD v33[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v34; // [rsp+70h] [rbp-40h]

  v8 = a2;
  v30 = *(_DWORD *)(a3 + 8);
  if ( v30 > 0x40 )
  {
    if ( v30 - (unsigned int)sub_C444A0(a3) <= 0x40 && !**(_QWORD **)a3 )
      goto LABEL_6;
  }
  else if ( !*(_QWORD *)a3 )
  {
    goto LABEL_6;
  }
  v10 = *((_BYTE *)a5 + 32);
  if ( v10 )
  {
    if ( v10 == 1 )
    {
      v33[0] = "sroa_idx";
      v34 = 259;
    }
    else
    {
      if ( *((_BYTE *)a5 + 33) == 1 )
      {
        v26 = (__int64 *)*a5;
        v28 = a5[1];
      }
      else
      {
        v26 = a5;
        v10 = 2;
      }
      v33[0] = v26;
      LOBYTE(v34) = v10;
      v33[1] = v28;
      v33[2] = "sroa_idx";
      HIBYTE(v34) = 3;
    }
  }
  else
  {
    v34 = 256;
  }
  v11 = (_BYTE *)sub_ACCFD0(*(__int64 **)(a1 + 72), a3);
  v12 = *(_QWORD **)(a1 + 72);
  v31[0] = v11;
  v13 = sub_BCB2B0(v12);
  v8 = sub_921130((unsigned int **)a1, v13, a2, v31, 1, (__int64)v33, 3u);
LABEL_6:
  v14 = *((_BYTE *)a5 + 32);
  if ( v14 )
  {
    if ( v14 != 1 )
    {
      if ( *((_BYTE *)a5 + 33) == 1 )
      {
        v27 = (_BYTE *)a5[1];
        a5 = (__int64 *)*a5;
        v29 = v27;
      }
      else
      {
        v14 = 2;
      }
      v31[0] = a5;
      LOBYTE(v32) = v14;
      v31[1] = v29;
      v31[2] = "sroa_cast";
      HIBYTE(v32) = 3;
      if ( a4 == *(_QWORD *)(v8 + 8) )
        return v8;
      goto LABEL_14;
    }
    v31[0] = "sroa_cast";
    v32 = 259;
  }
  else
  {
    v32 = 256;
  }
  if ( a4 == *(_QWORD *)(v8 + 8) )
    return v8;
LABEL_14:
  if ( *(_BYTE *)v8 > 0x15u )
  {
    v34 = 257;
    v8 = sub_B52190(v8, a4, (__int64)v33, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
      *(_QWORD *)(a1 + 88),
      v8,
      v31,
      *(_QWORD *)(a1 + 56),
      *(_QWORD *)(a1 + 64));
    v22 = *(unsigned int **)a1;
    v23 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
    while ( (unsigned int *)v23 != v22 )
    {
      v24 = *((_QWORD *)v22 + 1);
      v25 = *v22;
      v22 += 4;
      sub_B99FD0(v8, v25, v24);
    }
  }
  else
  {
    v16 = *(_QWORD *)(a1 + 80);
    v17 = *(__int64 (__fastcall **)(__int64, unsigned __int64, __int64))(*(_QWORD *)v16 + 144LL);
    if ( v17 == sub_B32D70 )
      v8 = sub_ADB060(v8, a4);
    else
      v8 = v17(v16, v8, a4);
    if ( *(_BYTE *)v8 > 0x1Cu )
    {
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
        *(_QWORD *)(a1 + 88),
        v8,
        v31,
        *(_QWORD *)(a1 + 56),
        *(_QWORD *)(a1 + 64));
      v18 = *(unsigned int **)a1;
      v19 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
      while ( (unsigned int *)v19 != v18 )
      {
        v20 = *((_QWORD *)v18 + 1);
        v21 = *v18;
        v18 += 4;
        sub_B99FD0(v8, v21, v20);
      }
    }
  }
  return v8;
}
