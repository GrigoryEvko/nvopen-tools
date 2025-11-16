// Function: sub_28EA500
// Address: 0x28ea500
//
__int64 __fastcall sub_28EA500(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdx
  __int64 v4; // rsi
  __int64 v5; // r14
  unsigned int v7; // eax
  __int64 v8; // rdi
  unsigned int v9; // r11d
  __int64 (__fastcall *v10)(__int64, __int64, __int64, unsigned __int8 *, _QWORD); // rax
  __int64 v11; // rax
  __int64 v12; // rcx
  int v13; // edi
  unsigned __int8 *v14; // r13
  __int64 v15; // rdi
  __int64 (__fastcall *v16)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v17; // r15
  __int64 v19; // rax
  __int64 v20; // rdx
  int v21; // r11d
  __int64 v22; // r13
  __int64 v23; // r15
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // r14
  __int64 v27; // r13
  __int64 v28; // rdx
  unsigned int v29; // esi
  int v30; // [rsp+0h] [rbp-B0h]
  unsigned __int8 v31; // [rsp+4h] [rbp-ACh]
  unsigned __int8 v32; // [rsp+6h] [rbp-AAh]
  unsigned int v33; // [rsp+10h] [rbp-A0h]
  _BYTE v34[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v35; // [rsp+40h] [rbp-70h]
  _BYTE v36[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v37; // [rsp+70h] [rbp-40h]

  v3 = *((unsigned int *)a2 + 2);
  v4 = *a2;
  if ( (_DWORD)v3 != 1 )
  {
    v5 = *(_QWORD *)(v4 + 8 * v3 - 8);
    v7 = v3 - 1;
    *((_DWORD *)a2 + 2) = v3 - 1;
    while ( 1 )
    {
      v12 = *(_QWORD *)(v5 + 8);
      v13 = *(unsigned __int8 *)(v12 + 8);
      if ( (unsigned int)(v13 - 17) <= 1 )
        LOBYTE(v13) = *(_BYTE *)(**(_QWORD **)(v12 + 16) + 8LL);
      v35 = 257;
      v14 = *(unsigned __int8 **)(v4 + 8LL * v7 - 8);
      *((_DWORD *)a2 + 2) = v7 - 1;
      if ( (_BYTE)v13 != 12 )
      {
        if ( *(_BYTE *)(a1 + 108) )
        {
          v5 = sub_B35400(a1, 0x6Cu, v5, (__int64)v14, v33, (__int64)v34, 0, v31, v32);
        }
        else
        {
          v8 = *(_QWORD *)(a1 + 80);
          v9 = *(_DWORD *)(a1 + 104);
          v10 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int8 *, _QWORD))(*(_QWORD *)v8 + 40LL);
          if ( (char *)v10 != (char *)sub_928A40 )
          {
            v11 = v10(v8, 18, v5, v14, v9);
            goto LABEL_9;
          }
          if ( *(_BYTE *)v5 <= 0x15u && *v14 <= 0x15u )
          {
            if ( (unsigned __int8)sub_AC47B0(18) )
              v11 = sub_AD5570(18, v5, v14, 0, 0);
            else
              v11 = sub_AABE40(0x12u, (unsigned __int8 *)v5, v14);
LABEL_9:
            if ( v11 )
            {
              v5 = v11;
              goto LABEL_11;
            }
            v9 = *(_DWORD *)(a1 + 104);
          }
          v30 = v9;
          v37 = 257;
          v19 = sub_B504D0(18, v5, (__int64)v14, (__int64)v36, 0, 0);
          v20 = *(_QWORD *)(a1 + 96);
          v21 = v30;
          v5 = v19;
          if ( v20 )
          {
            sub_B99FD0(v19, 3u, v20);
            v21 = v30;
          }
          sub_B45150(v5, v21);
          (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
            *(_QWORD *)(a1 + 88),
            v5,
            v34,
            *(_QWORD *)(a1 + 56),
            *(_QWORD *)(a1 + 64));
          v22 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
          if ( *(_QWORD *)a1 != v22 )
          {
            v23 = *(_QWORD *)a1;
            do
            {
              v24 = *(_QWORD *)(v23 + 8);
              v25 = *(_DWORD *)v23;
              v23 += 16;
              sub_B99FD0(v5, v25, v24);
            }
            while ( v22 != v23 );
          }
        }
LABEL_11:
        v7 = *((_DWORD *)a2 + 2);
        if ( !v7 )
          return v5;
        goto LABEL_12;
      }
      v15 = *(_QWORD *)(a1 + 80);
      v16 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v15 + 32LL);
      if ( v16 != sub_9201A0 )
        break;
      if ( *(_BYTE *)v5 <= 0x15u && *v14 <= 0x15u )
      {
        if ( (unsigned __int8)sub_AC47B0(17) )
          v17 = sub_AD5570(17, v5, v14, 0, 0);
        else
          v17 = sub_AABE40(0x11u, (unsigned __int8 *)v5, v14);
LABEL_21:
        if ( v17 )
          goto LABEL_22;
      }
      v37 = 257;
      v17 = sub_B504D0(17, v5, (__int64)v14, (__int64)v36, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
        *(_QWORD *)(a1 + 88),
        v17,
        v34,
        *(_QWORD *)(a1 + 56),
        *(_QWORD *)(a1 + 64));
      v26 = *(_QWORD *)a1;
      v27 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
      if ( *(_QWORD *)a1 != v27 )
      {
        do
        {
          v28 = *(_QWORD *)(v26 + 8);
          v29 = *(_DWORD *)v26;
          v26 += 16;
          sub_B99FD0(v17, v29, v28);
        }
        while ( v27 != v26 );
      }
LABEL_22:
      v7 = *((_DWORD *)a2 + 2);
      v5 = v17;
      if ( !v7 )
        return v5;
LABEL_12:
      v4 = *a2;
    }
    v17 = v16(v15, 17u, (_BYTE *)v5, v14, 0, 0);
    goto LABEL_21;
  }
  return *(_QWORD *)v4;
}
