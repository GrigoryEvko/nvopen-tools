// Function: sub_31EFFA0
// Address: 0x31effa0
//
void __fastcall sub_31EFFA0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  int v21; // r8d
  __int64 v22; // r13
  char v23; // bl
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned int v29; // ebx
  _QWORD *v30; // rax
  __int64 **v31; // rax
  _BYTE *v32; // rax
  _BYTE *v33; // rax
  __int64 v34; // rbx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // [rsp+0h] [rbp-70h]
  __int64 v38; // [rsp+8h] [rbp-68h]
  __int64 v39; // [rsp+10h] [rbp-60h]
  __int64 v41; // [rsp+20h] [rbp-50h]
  char v42; // [rsp+28h] [rbp-48h]
  char v43; // [rsp+28h] [rbp-48h]
  char v44; // [rsp+28h] [rbp-48h]
  int v45; // [rsp+28h] [rbp-48h]
  unsigned __int64 v46; // [rsp+30h] [rbp-40h] BYREF
  __int64 v47; // [rsp+38h] [rbp-38h]

  v37 = *(_QWORD *)(a2 + 8);
  v38 = *(_QWORD *)(v37 + 24);
  v6 = sub_9208B0(a1, v38);
  v47 = v7;
  v46 = v6;
  v8 = sub_CA1930(&v46);
  v42 = sub_AE5020(a1, v38);
  v9 = sub_9208B0(a1, v38);
  v47 = v10;
  v46 = 8 * (((1LL << v42) + ((unsigned __int64)(v9 + 7) >> 3) - 1) >> v42 << v42);
  if ( v8 == sub_CA1930(&v46) )
  {
    v11 = *(unsigned int *)(v37 + 32);
    if ( (_DWORD)v11 )
    {
      v12 = (unsigned int)v11;
      v13 = 0;
      v39 = v12;
      do
      {
        v41 = *(_QWORD *)(a2 + 8);
        v43 = sub_AE5020(a1, v41);
        v14 = sub_9208B0(a1, v41);
        v47 = v15;
        v46 = v13 * (((1LL << v43) + ((unsigned __int64)(v14 + 7) >> 3) - 1) >> v43 << v43);
        v16 = sub_CA1930(&v46);
        sub_31DB520(a3, v16, a4);
        v17 = (unsigned int)v13++;
        v18 = sub_AD69F0((unsigned __int8 *)a2, v17);
        sub_31E9900(a1, v18, a3, 0, 0, 0);
      }
      while ( v13 != v39 );
      v11 = *(unsigned int *)(v37 + 32);
    }
    v44 = sub_AE5020(a1, v38);
    v19 = sub_9208B0(a1, v38);
    v47 = v20;
    v46 = v11 * (((1LL << v44) + ((unsigned __int64)(v19 + 7) >> 3) - 1) >> v44 << v44);
    v21 = sub_CA1930(&v46);
  }
  else
  {
    v27 = sub_9208B0(a1, *(_QWORD *)(a2 + 8));
    v47 = v28;
    v46 = v27;
    v29 = sub_CA1930(&v46);
    v30 = (_QWORD *)sub_BD5C60(a2);
    v31 = (__int64 **)sub_BCCE00(v30, v29);
    v32 = (_BYTE *)sub_AD4C90(a2, v31, 0);
    v33 = (_BYTE *)sub_97B670(v32, a1, 0);
    v34 = (__int64)v33;
    if ( !v33 || *v33 != 17 )
      sub_C64ED0("Cannot lower vector global with unusual element type", 1u);
    sub_31DB520(a3, 0, a4);
    sub_31DA950(v34, a3);
    v35 = sub_9208B0(a1, *(_QWORD *)(a2 + 8));
    v47 = v36;
    v46 = (unsigned __int64)(v35 + 7) >> 3;
    v21 = sub_CA1930(&v46);
  }
  v22 = *(_QWORD *)(a2 + 8);
  v45 = v21;
  v23 = sub_AE5020(a1, v22);
  v24 = sub_9208B0(a1, v22);
  v47 = v25;
  v46 = (((unsigned __int64)(v24 + 7) >> 3) + (1LL << v23) - 1) >> v23 << v23;
  v26 = sub_CA1930(&v46) - v45;
  if ( v26 )
    sub_E99300(*(_QWORD ***)(a3 + 224), v26);
}
