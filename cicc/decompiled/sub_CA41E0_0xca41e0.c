// Function: sub_CA41E0
// Address: 0xca41e0
//
_QWORD *__fastcall sub_CA41E0(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 v3; // rax
  __int64 v4; // rbx

  if ( !byte_4F84F98 && (unsigned int)sub_2207590(&byte_4F84F98) )
  {
    v3 = sub_22077B0(336);
    v4 = v3;
    if ( v3 )
    {
      sub_CA23D0(v3, 1);
      qword_4F84FA0 = v4;
      _InterlockedAdd((volatile signed __int32 *)(v4 + 8), 1u);
    }
    else
    {
      qword_4F84FA0 = 0;
    }
    __cxa_atexit((void (*)(void *))sub_CA22E0, &qword_4F84FA0, &qword_4A427C0);
    sub_2207640(&byte_4F84F98);
  }
  v1 = qword_4F84FA0;
  *a1 = qword_4F84FA0;
  if ( v1 )
    _InterlockedAdd((volatile signed __int32 *)(v1 + 8), 1u);
  return a1;
}
