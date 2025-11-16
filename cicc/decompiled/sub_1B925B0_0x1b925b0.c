// Function: sub_1B925B0
// Address: 0x1b925b0
//
__int64 __fastcall sub_1B925B0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 *v5; // rdi
  __int64 v6; // rsi
  int v7; // eax
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  _QWORD *i; // rdx
  __int64 v15; // rdi
  unsigned int v16; // r12d
  __int64 v18; // [rsp+8h] [rbp-48h]
  _QWORD *v19; // [rsp+10h] [rbp-40h]
  int v20; // [rsp+1Ch] [rbp-34h]

  if ( *(_BYTE *)(a2 + 16) == 54 )
    v5 = *(__int64 **)a2;
  else
    v5 = **(__int64 ***)(a2 - 48);
  sub_1B8E090(v5, a3);
  v6 = sub_13A4950(a2);
  sub_1B8DFF0(a2);
  v7 = sub_1BF20B0(*(_QWORD *)(a1 + 320), v6);
  v8 = *(_QWORD *)(a1 + 320);
  v20 = v7;
  v9 = *(_QWORD *)(v8 + 504);
  if ( v9 == *(_QWORD *)(v8 + 496) )
    v10 = *(unsigned int *)(v8 + 516);
  else
    v10 = *(unsigned int *)(v8 + 512);
  v19 = (_QWORD *)(v9 + 8 * v10);
  v18 = *(_QWORD *)(a1 + 320);
  v11 = sub_15CC2D0(v8 + 488, a2);
  v12 = *(_QWORD *)(v18 + 504);
  if ( v12 == *(_QWORD *)(v18 + 496) )
    v13 = *(unsigned int *)(v18 + 516);
  else
    v13 = *(unsigned int *)(v18 + 512);
  for ( i = (_QWORD *)(v12 + 8 * v13); i != v11; ++v11 )
  {
    if ( *v11 < 0xFFFFFFFFFFFFFFFELL )
      break;
  }
  v15 = *(_QWORD *)(a1 + 328);
  if ( v11 == v19 )
  {
    v16 = sub_14A34A0(v15);
    if ( v20 >= 0 )
      return v16;
LABEL_14:
    v16 += sub_14A3380(*(_QWORD *)(a1 + 328));
    return v16;
  }
  v16 = sub_14A34D0(v15);
  if ( v20 < 0 )
    goto LABEL_14;
  return v16;
}
