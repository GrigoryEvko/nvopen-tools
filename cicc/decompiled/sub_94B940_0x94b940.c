// Function: sub_94B940
// Address: 0x94b940
//
__int64 __fastcall sub_94B940(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 result; // rax
  __int64 v6; // rax
  _BYTE *v7; // r12
  int v8; // eax
  __int64 v9; // rdi
  __int64 (__fastcall *v10)(__int64, _BYTE *, _BYTE *); // rax
  __int64 v11; // r14
  __int64 v12; // rax
  char v13; // al
  int v14; // ebx
  __int64 v15; // rax
  int v16; // r9d
  __int64 v17; // r12
  unsigned int *v18; // r14
  unsigned int *v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rax
  unsigned int *v23; // r12
  unsigned int *v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // rax
  int v28; // r14d
  __int64 v29; // rax
  __int64 v30; // r9
  __int64 v31; // r12
  unsigned int *v32; // rbx
  unsigned int *i; // r13
  __int64 v34; // rdx
  __int64 v35; // rsi
  __int64 v36; // [rsp-10h] [rbp-130h]
  __int64 v37; // [rsp+18h] [rbp-108h]
  __int64 v38; // [rsp+20h] [rbp-100h]
  __int64 v39; // [rsp+28h] [rbp-F8h]
  unsigned __int16 v42; // [rsp+4Eh] [rbp-D2h]
  int v43; // [rsp+50h] [rbp-D0h]
  __int64 v44; // [rsp+58h] [rbp-C8h]
  char v45[32]; // [rsp+60h] [rbp-C0h] BYREF
  __int16 v46; // [rsp+80h] [rbp-A0h]
  char v47[32]; // [rsp+90h] [rbp-90h] BYREF
  __int16 v48; // [rsp+B0h] [rbp-70h]
  _BYTE v49[32]; // [rsp+C0h] [rbp-60h] BYREF
  __int16 v50; // [rsp+E0h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 > 1 )
  {
    v27 = sub_AA4E30(*(_QWORD *)(a1 + 96));
    v28 = (unsigned __int8)sub_AE5020(v27, *(_QWORD *)(a2 + 8));
    v50 = 257;
    v29 = sub_BD2C40(80, unk_3F10A10);
    v31 = v29;
    if ( v29 )
    {
      sub_B4D3C0(v29, a2, a3, 0, v28, v30, 0, 0);
      v30 = v36;
    }
    result = (*(__int64 (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 136)
                                                                                          + 16LL))(
               *(_QWORD *)(a1 + 136),
               v31,
               v49,
               *(_QWORD *)(a1 + 104),
               *(_QWORD *)(a1 + 112),
               v30);
    v32 = *(unsigned int **)(a1 + 48);
    for ( i = &v32[4 * *(unsigned int *)(a1 + 56)]; i != v32; result = sub_B99FD0(v31, v35, v34) )
    {
      v34 = *((_QWORD *)v32 + 1);
      v35 = *v32;
      v32 += 4;
    }
  }
  else
  {
    v39 = sub_BCB2D0(*(_QWORD *)(a1 + 40));
    v38 = *(_QWORD *)(v4 + 24);
    result = *(unsigned int *)(v4 + 32);
    if ( (_DWORD)result )
    {
      v37 = *(unsigned int *)(v4 + 32);
      v44 = 0;
      do
      {
        v6 = sub_AD64C0(v39, v44, 0);
        v48 = 257;
        v7 = (_BYTE *)v6;
        v8 = sub_94B2B0((unsigned int **)(a1 + 48), v38, a3, v44, (__int64)v47);
        v9 = *(_QWORD *)(a1 + 128);
        v43 = v8;
        v46 = 257;
        v10 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *))(*(_QWORD *)v9 + 96LL);
        if ( v10 == sub_948070 )
        {
          if ( *(_BYTE *)a2 > 0x15u || *v7 > 0x15u )
          {
LABEL_15:
            v50 = 257;
            v22 = sub_BD2C40(72, 2);
            v11 = v22;
            if ( v22 )
              sub_B4DE80(v22, a2, v7, v49, 0, 0);
            (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
              *(_QWORD *)(a1 + 136),
              v11,
              v45,
              *(_QWORD *)(a1 + 104),
              *(_QWORD *)(a1 + 112));
            v23 = *(unsigned int **)(a1 + 48);
            v24 = &v23[4 * *(unsigned int *)(a1 + 56)];
            while ( v24 != v23 )
            {
              v25 = *((_QWORD *)v23 + 1);
              v26 = *v23;
              v23 += 4;
              sub_B99FD0(v11, v26, v25);
            }
            goto LABEL_9;
          }
          v11 = sub_AD5840(a2, v7, 0);
        }
        else
        {
          v11 = v10(v9, (_BYTE *)a2, v7);
        }
        if ( !v11 )
          goto LABEL_15;
LABEL_9:
        v12 = sub_AA4E30(*(_QWORD *)(a1 + 96));
        v13 = sub_AE5020(v12, *(_QWORD *)(v11 + 8));
        v14 = v42;
        v50 = 257;
        LOBYTE(v14) = v13;
        v42 = v14;
        v15 = sub_BD2C40(80, unk_3F10A10);
        v17 = v15;
        if ( v15 )
          sub_B4D3C0(v15, v11, v43, 0, v14, v16, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
          *(_QWORD *)(a1 + 136),
          v17,
          v49,
          *(_QWORD *)(a1 + 104),
          *(_QWORD *)(a1 + 112));
        v18 = *(unsigned int **)(a1 + 48);
        v19 = &v18[4 * *(unsigned int *)(a1 + 56)];
        while ( v19 != v18 )
        {
          v20 = *((_QWORD *)v18 + 1);
          v21 = *v18;
          v18 += 4;
          sub_B99FD0(v17, v21, v20);
        }
        result = ++v44;
      }
      while ( v44 != v37 );
    }
  }
  return result;
}
