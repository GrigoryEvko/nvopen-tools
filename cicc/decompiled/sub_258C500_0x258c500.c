// Function: sub_258C500
// Address: 0x258c500
//
__int64 __fastcall sub_258C500(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  int v4; // edx
  __int64 v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // rdx
  int v8; // r11d
  unsigned int i; // eax
  _QWORD *v10; // r10
  unsigned int v11; // eax
  unsigned __int8 *v12; // rax
  unsigned __int8 *v13; // r14
  __int64 result; // rax
  __int64 v15; // rbx
  unsigned __int8 *v16; // rax
  __int64 v17; // [rsp-10h] [rbp-40h]

  v2 = a2;
  v4 = *(_DWORD *)(a2 + 56);
  v5 = *(_QWORD *)(a2 + 40);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a1 + 72);
    v7 = (unsigned int)(v4 - 1);
    a2 = *(_QWORD *)(a1 + 80);
    v8 = 1;
    for ( i = v7
            & (((unsigned int)a2 >> 9)
             ^ ((unsigned int)a2 >> 4)
             ^ (16 * (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)))); ; i = v7 & v11 )
    {
      v10 = (_QWORD *)(v5 + ((unsigned __int64)i << 6));
      if ( v6 == *v10 && a2 == v10[1] )
        break;
      if ( unk_4FEE4D0 == *v10 && qword_4FEE4D8 == v10[1] )
        goto LABEL_7;
      v11 = v8 + i;
      ++v8;
    }
    return (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 128LL))(
             a1,
             a2,
             v7,
             v6,
             unk_4FEE4D0,
             qword_4FEE4D8);
  }
  else
  {
LABEL_7:
    v12 = (unsigned __int8 *)sub_250D070((_QWORD *)(a1 + 72));
    v13 = sub_BD3990(v12, a2);
    result = *v13;
    if ( (_BYTE)result != 5 && (unsigned __int8)result <= 0x15u )
    {
      v15 = sub_25096F0((_QWORD *)(a1 + 72));
      v16 = (unsigned __int8 *)sub_2509740((_QWORD *)(a1 + 72));
      sub_258BA20(a1, v2, (_BYTE *)(a1 + 88), (unsigned __int64)v13, v16, 3, v15);
      *(_BYTE *)(a1 + 104) = *(_BYTE *)(a1 + 105);
      return v17;
    }
  }
  return result;
}
